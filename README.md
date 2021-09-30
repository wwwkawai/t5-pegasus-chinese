# t5-pegasus-chinese
基于GOOGLE T5中文生成式模型的摘要生成/指代消解，支持batch批量生成，多进程

**如果你想了解自己是否需要本Git，请看如下几点介绍（重点）：**
1. 模型可部署在CPU/GPU，均测试可用
2. 基于谷歌t5的中文生成式预训练模型
3. 集成了中文摘要生成、指代消解等生成任务语料，开箱即用
4. 基于PyTorch
5. 支持多张显卡DataParallel
6. 支持批量推理/生成，提速明显
7. 支持多进程，进一步提速优化

**本 Git 如何运行：**  
1. 安装所需Python库  
    - transformers==4.3.3  
    - tokeniziers==0.10.3  
    - keras4bert==0.15.6
    - torch==1.7.0或者1.8.0
    - Keras==2.3.1
    - tensorflow-gpu==1.15.0
    - jieba
    - rouge
    - tqdm
2. t5-pegasus模型放在 t5_pegasus_pretain目录下，目录下三个文件：
   - pytorch_model.bin
   - bert_config.json
   - vocab.txt  

    预训练模型下载地址：
    - Base版本：https://pan.baidu.com/s/1lQ9Dt9wZDO3IgiCL9tP-Ug 提取码: 3sfn
    - Small版本：链接: https://pan.baidu.com/s/1bXRVWnDyAck9VfSO9_1oJQ 提取码: qguk

    解压后，按上面说的放在对应目录下，文件名称确认无误即可。
3. 命令行执行
   - 训练finetune
        ```bash
        python train_with_finetune.py
        ```
   - 预测generate
        ```bash
        # 文件预测
        python predict_with_generate.py
        # 单条预测 GPU
        > import torch
        > from
        # 单条预测 CPU
        > import torch
        > from
        ```

**t5-pegasus模型的细节，以便了解它为什么能在摘要任务中有效:**

**实验结果:**


**如对本Git内容存有疑问或建议，欢迎在issue区或者邮箱isguanjing@126.com与我联系。**