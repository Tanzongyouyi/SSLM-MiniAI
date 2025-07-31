import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
import json
import tiktoken

# 设置环境变量和线程数
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
torch.set_num_threads(12)
torch.manual_seed(1024)

@dataclass
class GPTConfig:
    block_size: int = 384
    batch_size: int = 8
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 50257

class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size
        # 修复：将tril注册为buffer，这样它会被保存到state_dict中
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)
        # 修复：使用动态生成的tril，而不是固定的
        mask = self.tril[:seq_len, :seq_len]
        weight = weight.masked_fill(mask == 0, float('-inf')) / math.sqrt(self.head_size)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class MyDataset(Dataset):
    def __init__(self, path, block_size=384):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.eos_token = self.enc.encode("Ċ", allowed_special={"Ċ"})[0]
        
        self.encoded_data = []
        self.max_lines = 500
        raw_data = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i+self.block_size+1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        #将文本编码为token IDs
        return self.enc.encode(text)

    def decode(self, ids):
        #将token IDs解码为文本
        return self.enc.decode(ids)

def main():
    # 使用新数据集
    train_dataset = MyDataset('newdata.jsonl')
    
    total_len = len(train_dataset)
    if total_len < 2:
        train_size = total_len
        val_size = 0
    else:
        train_size = max(1, int(total_len * 0.9))
        val_size = total_len - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=12)  # 减少num_workers
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=12)  # 减少num_workers
    
    # 加载预训练模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(GPTConfig()).to(device)
    
    # 加载预训练权重
    checkpoint = torch.load('checkpoints/model_epoch_10.pt', map_location=device)
    
    # 修复：strict=False允许部分加载，忽略不匹配的键
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    def train(model, optimizer, scheduler, train_loader, device):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train", ncols=100)
        for batch_idx, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return total_loss

    def eval(model, val_loader, device):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, total=len(val_loader), desc="Val", ncols=100)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, targets=y)
                val_loss += loss.item()
        return val_loss

    os.makedirs('checkpoints', exist_ok=True)
    
    # 微调3-5轮
    num_epochs = 3  # 可根据需要调整为3-5
    
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, scheduler, train_loader, device)
        val_loss = eval(model, val_loader, device) if len(val_loader) > 0 else 0
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}', end='')
        if len(val_loader) > 0:
            print(f', Val Loss: {val_loss/len(val_loader):.4f}')
        else:
            print(', Val Loss: N/A')
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+11}.pt')  # 保存为epoch11开始

if __name__ == '__main__':
    main()
