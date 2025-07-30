import time
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from main import GPT, GPTConfig, MyDataset
import uvicorn

app = FastAPI()

# 加载模型和tokenizer
model = GPT(GPTConfig())
checkpoint = torch.load('checkpoints/model_epoch_6.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = MyDataset('data.jsonl')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        input_ids = tokenizer.encode(data)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        output_ids = input_ids.copy()
        start_time = time.time()
        token_count = 0
        for _ in range(50):
            with torch.no_grad():
                out = model.generate(torch.tensor([output_ids], dtype=torch.long), 1)[0].tolist()
            new_token = out[-1]
            output_ids.append(new_token)
            token_count += 1
            elapsed = time.time() - start_time
            tokens_per_s = token_count / elapsed if elapsed > 0 else 0
            await websocket.send_json({
                "token": tokenizer.decode([new_token]),
                "tokens_per_s": tokens_per_s
            })
            if new_token == tokenizer.eos_token:
                break
        await websocket.send_json({"token": "[END]", "tokens_per_s": tokens_per_s})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11451)