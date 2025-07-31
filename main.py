import torch
import os
os.environ[OMP_NUM_THREADS] = 12
os.environ[MKL_NUM_THREADS] = 12
torch.set_num_threads(12)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

import math

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    block_size: int = 384   # 这里其实应该是文本的最大长度（ max_seq_len）
    batch_size: int = 8
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768    # n_embd 也叫 hidden_dim, hiden_size, 这里我同时设置了和 embed_dim 一样
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    # # tiktoken 使用的是 GPT-2 的词表，大约有 50257 个token
    vocab_size: int = 50257

class SingleHeadAttention(nn.Module):
    # 单头注意力机制
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size

        # 尝试学习新的写法，tril 通过 register_buffer 注册
        # 因为不用计算 梯度，所以节约内存和显存，速度也更快
        self.register_buffer(
            'tril', 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)   # @ 就是 torch.matmul 的简化写法
        # 一定要在 softmax 前除以 sqrt(head_size)
        weight = weight.masked_fill(
            self.tril[:seq_len, :seq_len] == 0, 
            float('-inf')
        ) / math.sqrt(self.head_size)  # 这里的 hidden_size 其实是 head_size，因为是单头
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads], 
            dim=-1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    # 实际上就是 MLP
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


# 接下来就是一个完整的 Block

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# 以后会讲  MLA ,  MOE, DPO 完全手写
# 完整的  GPT model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size  # 新增，供generate使用
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # linear (4 -> 8)； weight shape 是记上是 8 * 4，
        # 所以 embedding weight 和 lm_head weight 是共享的
        # 这里学习一下 tie weight。
        # 这是为了减少参数，加快训练；（现在 25的 SLM 很多都这样做了，注意⚠️）
        # self.token_embedding_table.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 这里使用的是正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx 是输入的 token ids
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)

        # seq 长度是这次输入的最大长度
        pos_emb = self.position_embedding_table(
            # 要确保 位置编码和输入的 idx 在同一个设备上
            torch.arange(seq_len, device=idx.device)
        )
        # 有一个经典题目：为什么 embedding 和 position 可以相加？
        x = token_emb + pos_emb   # shape is (batch, seq_len, n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)   # shape is (batch, seq_len, vocab_size)
        
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后 block_size 个token
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 获取预测
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx



# 写一个 dataset，为了 Dataloader 准备
class MyDataset(Dataset):
    def __init__(self, path, block_size=384):
        # 我的数据在 /root/fs/mobvoi_seq_monkey_general_open_corpus.jsonl 中，
        # 读取前 1000 行
        import tiktoken
        self.enc = tiktoken.get_encoding(gpt2)
        self.block_size = block_size


        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        import json

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
        
        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i+self.block_size+1]
            # 如果长度不够，用 eos_token 填充
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
    # train data
    train_dataset = MyDataset('data.jsonl')

    # split traindataset to train and val，适配极小数据集，保证训练集和验证集都不会为0
    total_len = len(train_dataset)
    if total_len < 2:
        train_size = total_len
        val_size = 0
    else:
        train_size = max(1, int(total_len * 0.9))
        val_size = total_len - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=12)
    model = GPT(GPTConfig())
    device = cuda if torch.cuda.is_available() else cpu
    model = model.to(device)

    # 打印模型一共有多少参数
    total_params = sum(p.numel() for p in model.parameters())
    print("fTotal parameters:", {total_params / 1e6},"M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # 设置 cosine 学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # 训练循环
    def train(model, optimizer, scheduler, train_loader, val_loader, device):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=Train, ncols=100)
        for batch_idx, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        return total_loss

    def eval(model, val_loader, device):
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, total=len(val_loader), desc=Val, ncols=100)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, targets=y)
                val_loss += loss.item()
        return val_loss

    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(30):
        train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
        val_loss = eval(model, val_loader, device) if len(val_loader) > 0 else 0
        if len(train_loader) > 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}', end='')
        else:
            print(f'Epoch: {epoch}, Train Loss: N/A', end='')
        if len(val_loader) > 0:
            print(f', Val Loss: {val_loss/len(val_loader):.4f}')
        else:
            print(', Val Loss: N/A')
        # 保存模型
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')

if __name__ == '__main__':
    main()
    