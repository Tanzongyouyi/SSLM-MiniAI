## ⭐SSLM-MiniAI⭐ 一个不太聪明的中文小模型

### 1.概述
这可能是我从2020年以来第一个算是拿得出手的项目，本模型的训练代码主要来源于 [LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero) ，感谢 [bbruceyuan](https://github.com/bbruceyuan) 的无私奉献。本仓库中的 [main.py](https://github.com/Tanzongyouyi/SSLM-MiniAI/blob/main/main.py) 是经过定制的训练代码，它对小型数据集和多核CPU有较好的支持，能够充分利用核心，以供无卡或集成显卡用户使用。

本模型仅120M参数，数据集为 deepseek-r1:1.5b 的[对话记录](https://github.com/Tanzongyouyi/SSLM-MiniAI/blob/main/data.jsonl)，但原链接已经遗失，本模型(v1.1)使用前1000行，每行取前384个字符，第8轮的最终结果:Train Loss = 1.8，Val Loss = 2.3，后来就过拟合了，因此没有继续训练。

### 2.文件结构
SSLM-MiniAI</br>
|   data.jsonl</br>
|   data.txt</br>
|   demo.py # 本模型的控制台对话工具</br>
|   main.py</br>
|   README.md</br>
|   scissors.py # 数据处理工具，可将data.txt整合为符合格式要求的data.jasonl</br>
|   </br>
+---checkpoints</br>
|-------model_epoch_7.pt</br>

### 3.使用
1.运行以下命令以下载支持库：

```bash
pip install torch fastapi uvicorn websockets tqdm tiktoken dataclasses
```

2.准备好数据后，根据自身设备配置修改main.py</br>
3.运行main.py，默认训练30轮，模型文件保存在checkpoints文件夹</br>
4.运行demo.py，即可控制台对话</br>

### 4.碎碎念

要不是没有独显，我早就把data.jsonl的所有数据喂给AI了，也不至于被cpu核心数和内存限制手脚。建议大家还是花个一千买张卡，体验会好得多。希望以后能弄一张2080Ti 22G来跑模型。运行过程中遇到问题记得提issue，别忘了给这个仓库点个star哦😘

### 5.To Do

1.更新v1.0 √</br>
2.更新v1.1 √</br>
3.找到最佳组合并更新v2.0</br>
4.买2080Ti 22G
