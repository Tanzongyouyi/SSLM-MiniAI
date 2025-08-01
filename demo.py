import torch
from weitiao import GPT, GPTConfig, MyDataset

# 加载模型
checkpoint = torch.load('checkpoints/model_epoch_20.pt', map_location='cpu')
model = GPT(GPTConfig())
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载tokenizer
tokenizer = MyDataset('自我认知.jsonl')  # 只用encode/decode，不用数据

while True:
    user_input = input("你：")
    if user_input.strip().lower() in ['exit', 'quit']:
        break
    # 编码
    input_ids = tokenizer.encode(user_input)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    # 生成
    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_new_tokens=50)[0].tolist()
    # 解码
    response = tokenizer.decode(output_ids[len(input_ids):])  # 只取新生成部分
    print("模型：", response)