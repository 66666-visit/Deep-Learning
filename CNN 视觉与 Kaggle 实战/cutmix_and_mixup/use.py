# 假设你在 train.py 里
from utils_aug import cutmix_data, mixup_data, MixCriterion
import torch
import numpy as np
from torch import nn
device = 'cpu'
train_loader = []
class model:
    def __init__(self):
        self.x = 1
# 1. 初始化 Loss
base_criterion = torch.nn.CrossEntropyLoss()
mix_criterion_func = MixCriterion(base_criterion) # 实例化混合 Loss

# 2. 训练循环内
for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    
    # === 增强逻辑开始 ===
    r = np.random.rand(1)
    
    if r < 0.5: 
        # 50% 概率触发 CutMix (推荐首选)
        images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
        outputs = model(images)
        loss = mix_criterion_func(outputs, targets_a, targets_b, lam)
        
    elif r < 0.0: 
        # 如果你想混用 Mixup，可以调整这里的概率
        # 比如 r > 0.5 and r < 0.7 触发 Mixup
        images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)
        outputs = model(images)
        loss = mix_criterion_func(outputs, targets_a, targets_b, lam)
        
    else:
        # 正常训练 (不增强)
        outputs = model(images)
        loss = base_criterion(outputs, labels)
    # === 增强逻辑结束 ===
