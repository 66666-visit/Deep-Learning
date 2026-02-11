# 🚀 21天 深度学习硬核突击计划 (v2.0 竞赛加强版)

**核心原则**：
1.  **不求甚解**：数学公式看不懂就跳过，代码跑通是第一位。
2.  **以赛代练**：不要空学理论，一切为了在 Kaggle 上拿分。
3.  **拥抱标准**：如果原仓库代码太乱或为空，直接向 Gemini 索要“工业界标准模板”。

---

## 📅 第一阶段：PyTorch 扫盲与工程地基（第 1-5 天）
> **目标**：脱离文档，徒手写出 `Dataset` 和 `Train Loop`。

### Day 1: 搞定数据（Kaggle 第一关）
* **对应文件**：
    * `103_Pytorch加载数据.ipynb`
    * `107_Dataloader使用.ipynb`
    * `232_数据增广.ipynb`
* **核心任务**：
    * 学会写自定义 `Dataset` 类（继承 `torch.utils.data.Dataset`）。
    * **实战**：读取一个带有 CSV 索引的图片文件夹，而不是标准的 ImageFolder。

### Day 2: 搭建模型积木
* **对应文件**：
    * `108_nn.Module模块使用.ipynb`
    * `114_搭建小实战.ipynb`
* **核心任务**：
    * 理解 `__init__` 里定义层，`forward` 里连接层。
    * 搞懂 tensor 在网络里流动时的 `shape` 变化。

### Day 3: 训练的核心（Loss & Optimizer）
* **对应文件**：
    * `115_损失函数.ipynb`
    * `116_优化器.ipynb`
* **核心任务**：
    * **避坑**：搞清楚 CrossEntropyLoss 的输入是 Logits（未归一化数值）而不是 Probability。

### Day 4: 完整流水线（背诵全文）
* **对应文件**：
    * `119_完整模型训练套路.ipynb` (⭐️ **重点背诵**)
    * `121_完整模型验证套路.ipynb`
* **核心任务**：
    * **整理模板**：将代码整理成你的“万能模板” (Data -> Model -> Loss -> Loop)。
    * **要求**：以后打比赛，这套代码要能直接复制粘贴，改改数据路径就能跑。

### Day 5: 模型保存与 GPU
* **对应文件**：
    * `118_网络模型保存与读取.ipynb`
    * `120_利用GPU训练.ipynb`
* **核心任务**：
    * 学会保存训练好的模型权重，并能重新加载继续训练（Resume Training）。

---

## 👁️ 第二阶段：CNN 视觉与 Kaggle 实战（第 6-13 天）
> **目标**：攻克 CNN，掌握**分类、检测、分割**三大任务，熟练使用竞赛 Trick。

### Day 6: 卷积与池化（CNN 基础）
* **对应文件**：
    * `216_卷积层.ipynb`
    * `219_池化层.ipynb`
* **重点**：理解 **通道 (Channel)** 和 **特征图 (Feature Map)** 的概念。

### Day 7: 经典网络架构
* **对应文件**：
    * `222_VGG.ipynb`
    * `224_GoogLeNet.ipynb`
* **重点**：快速浏览，学习大神如何堆叠 Block。

### Day 8: 残差网络（ResNet - 必修）
* **对应文件**：
    * `226_残差神经网络ResNet.ipynb` (⭐️ **核心**)
    * `225_批量归一化(BatchNorm).ipynb`
* **重点**：**ResNet 是基石**。理解 Residual Block 如何解决梯度消失，让你能训练几百层的网络。

### Day 9: 微调（Transfer Learning - 拿分核心）
* **对应文件**：
    * `233_微调.ipynb` (⭐️ **核心**)
* **核心任务**：
    * 学会加载 `torchvision.models` 里的预训练权重。
    * **冻结层 (Freeze)**：冻结特征提取层，只训练分类头 (Classifier Head)。

### 🔥 Day 10: 图像分类实战 + 进阶增强 (CutMix)
* **对应文件**：
    * `234_CIFAR10.ipynb` 或 `235_狗的品种识别.ipynb`
* **⚠️ 额外补充 (原仓库无)**：
    * **CutMix / Mixup**：不要只做旋转裁剪。学会把两张图混合在一起训练，这是防止过拟合的大杀器。
    * **update** 2026.2.8更新了**CutMix/Mixup**模版代码

### 🔥 Day 11: 竞赛大杀器——K-Fold 与 Scheduler
* **对应文件**：
    * `238_树叶分类竞赛技术总结.ipynb` (⚠️ **注意：此文件看看即可，需补充代码**)
* **⚠️ 额外补充 (稳分核心)**：
    * **K-Fold (K折交叉验证)**：学会用 `sklearn.model_selection.StratifiedKFold` 把数据切 5 份，跑 5 个模型，最后取平均。
    * **Scheduler (学习率调度)**：学会用 `CosineAnnealingLR` 让学习率像波浪一样下降，而不是死板的一条直线。
    * *这是原仓库缺失的部分，务必重点攻克。*
      **update** 2026.2.9更新了**K-Fold/CosineAnnelingLR**模版代码

### Day 12: 目标检测基础
* **对应文件**：
    * `236_物体检测和数据集.ipynb`
    * `237_锚框.ipynb`
    * `239_检测算法.ipynb`
* **目标**：理解检测任务的输入（图片）和输出（坐标框 + 类别）是什么格式。

### 🔥 Day 13: 图像分割与 U-Net (新增重磅)
* **对应文件**：
    * `244_全连接卷积神经网络FCN.ipynb` (作为前置学习)
* **⚠️ 额外补充 (核心)**：
    * **U-Net 架构**：Kaggle 分割任务王者。
    * 掌握 **Skip Connection (跳跃连接)**：如何把浅层特征拼接到深层，恢复图像细节。
      **update** 2026.2.11 更新U-net模版代码

---

## ⏳ 第三阶段：LSTM 与 序列模型（第 14-19 天）
> **目标**：解决时序预测或文本分类问题，理解 Attention 机制。

### Day 14: RNN 基础
* **对应文件**：
    * `249_循环神经网络RNN.ipynb`
    * `247_文本预处理.ipynb`
* **重点**：理解时间步 (Time Step) 和 Hidden State。

### Day 15: LSTM & GRU（重点攻克）
* **对应文件**：
    * `251_GRU.ipynb`
    * `252_LSTM.ipynb`
* **核心任务**：
    * 细看 PyTorch 的 `nn.LSTM` 参数：`input_size`, `hidden_size`, `num_layers`, `batch_first`。一定要搞懂输入输出的维度形状。

### Day 16: Seq2Seq 架构
* **对应文件**：
    * `256_编码器解码器.ipynb`
    * `257_seq2seq.ipynb`
* **重点**：编码器-解码器思想，这是生成式任务的祖师爷。

### Day 17: 注意力机制（Attention）
* **对应文件**：
    * `259_注意力机制.ipynb`
    * `261_seq2seq注意力.ipynb`
* **重点**：理解 Attention 是如何让模型“聚焦”到输入的某个部分的。

### Day 18: Transformer（现在的王）
* **对应文件**：
    * `263_Transformer.ipynb`
* **核心任务**：
    * 虽然主攻 LSTM，但必须看懂 Transformer 的 **Self-Attention** 公式。现在的时序比赛（如时间序列预测）经常用 Transformer 变体。

### Day 19: 竞赛总结与多模态
* **对应文件**：
    * `266_目标检测竞赛总结.ipynb`
* **任务**：复盘前面的模型，了解一下 CNN + RNN 的组合（例如视频分类）。

---

## 🛠️ 第四阶段：查漏补缺与终极复盘（第 20-21 天）

### Day 20: 进阶调优
* **对应文件**：
    * `211_丢弃法(Dropout).ipynb`
    * `212_数值稳定性.ipynb`
* **内容**：防止过拟合的手段，比赛后期调优必用。

### Day 21: 终极代码默写
* **复习对象**：
    * `119_完整模型训练套路.ipynb`
    * `233_微调.ipynb`
* **挑战**：不看任何资料，在一个空白 `.py` 文件里写出：
    1.  一个带 ResNet 微调的 Model 类。
    2.  一个包含 K-Fold 循环的完整训练流程。
    3.  U-Net 的 Skip Connection 拼接代码。
