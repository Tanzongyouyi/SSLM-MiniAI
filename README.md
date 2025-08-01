## ⭐SSLM-MiniAI⭐ 一个不太聪明的中文小模型

### 喜报！模型Val Loss已突破1.5！

### 1.概述
这可能是我从2020年以来第一个拿得出手的项目，本模型的训练代码主要来源于 [LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero) ，感谢 [bbruceyuan](https://github.com/bbruceyuan) 的无私奉献。本仓库中的 [main.py](https://github.com/Tanzongyouyi/SSLM-MiniAI/blob/main/main.py) 是经过定制的训练代码，它对小型数据集和多核CPU有较好的支持，能够充分利用核心，以供无卡或集成显卡用户使用。

本模型仅120M参数，数据集为 deepseek-r1:1.5b 的[对话记录](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/file/view/master/r1_mix_1024.jsonl?id=68909&status=2)，本模型(v2.1)使用前2000行，每行取前384个字符，最终结果:Epoch: 20, Train Loss: 0.5599, Val Loss: 1.4080，已经可以输出一些无错误句子。

对于 i5-1240p 这颗12核16线程的cpu，训练一轮大约需要20分钟。近期不更新v3.0，得去沉淀一下。

### 2.文件结构
SSLM-MiniAI</br>
|   data.jsonl # 预训练数据</br>
|   自我认知.jsonl</br>
|   data.txt # 预处理数据</br>
|   demo.py # 本模型的控制台对话工具</br>
|   main.py # 预训练脚本</br>
|   newdata.jsonl # 微调数据</br>
|   weitiao.py # 微调脚本</br>
|   README.md</br>
|   scissors.py # 数据处理工具，可将data.txt整合为符合格式要求的data.jsonl</br>
|   </br>
+---checkpoints</br>
|-------model_epoch_20.pt # 模型文件</br>

### 3.使用
1.运行以下命令以下载支持库：

```bash
pip install torch fastapi uvicorn websockets tqdm tiktoken dataclasses
```

2.准备好数据后，根据自身设备配置修改main.py</br>
3.运行main.py，默认训练30轮，模型文件保存在checkpoints文件夹</br>
4.运行demo.py，即可控制台对话</br>
5.准备newdata.jsonl，运行weitiao.py即可微调模型，默认训练5轮

### 4.碎碎念

要不是没有独显，我早就把所有数据一次性喂给AI了，也不至于被cpu核心数和内存限制手脚。建议大家还是花个一千买张卡，体验会好得多。希望以后能弄一张2080Ti 22G来跑模型。运行过程中遇到问题记得提issue，别忘了给这个仓库点个star哦😘

### 5.To Do

1.更新v1.0 √</br>
2.更新v1.1 √</br>
3.更新v2.0 √</br>
4.买2080Ti 22G
