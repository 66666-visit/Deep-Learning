import numpy as np
import torch

def rand_bbox(size, lam):
    """
    生成 CutMix 剪切框的坐标
    Args:
        size: data.size() -> [batch_size, channels, width, height]
        lam: Lambda 参数 (保留原图的比例)
    Returns:
        bbx1, bby1, bbx2, bby2 (裁剪框坐标)
    """
    W = size[2]
    H = size[3]
    # 根据 lambda 计算剪切掉的宽高 (cut_ratio)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机生成中心点 cx, cy
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 限制坐标在图片范围内 (Clip)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(data, targets, alpha=1.0):
    """
    执行 CutMix 数据增强
    Args:
        data: 输入图片 Batch
        targets: 标签 Batch
        alpha: Beta 分布参数，通常取 1.0
    Returns:
        mixed_data: 混合后的图片
        targets_a: 原图标签
        targets_b: 混合图标签
        lam: 最终修正后的混合比例
    """
    indices = torch.randperm(data.size(0)).to(data.device)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    # 从 Beta 分布采样 lambda
    lam = np.random.beta(alpha, alpha)
    
    # 生成剪切框
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    
    # 【核心操作】将 shuffled_data 的区域贴到 data 上
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
    # 【绝对正确修正】由于边界裁剪，实际剪切面积可能与采样的 lambda 不同
    # 必须重新计算准确的 lambda，否则 Loss 计算会不准
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    return data, targets, shuffled_targets, lam

def mixup_data(data, targets, alpha=1.0):
    """
    执行 Mixup 数据增强
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size(0)
    indices = torch.randperm(batch_size).to(data.device)

    # 简单的线性加权混合
    mixed_data = lam * data + (1 - lam) * data[indices, :]
    targets_a, targets_b = targets, targets[indices]
    
    return mixed_data, targets_a, targets_b, lam

class MixCriterion(torch.nn.Module):
    """
    混合 Loss 计算器
    公式: Loss = λ * Loss(a) + (1-λ) * Loss(b)
    """
    def __init__(self, criterion):
        super(MixCriterion, self).__init__()
        self.criterion = criterion

    def forward(self, preds, targets_a, targets_b, lam):
        return lam * self.criterion(preds, targets_a) + (1 - lam) * self.criterion(preds, targets_b)