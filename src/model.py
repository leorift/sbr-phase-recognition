# -*- coding: utf-8 -*-
"""
MS-TCN++ 多阶段时序卷积分割网络：
- 双扩展层 DDL：小扩展率分支捕获局部突变（气泡出现/搅拌启动），
  大扩展率分支捕获长程趋势（水位升降/泥柱沉降），逐层扩展率指数增长；
- 第 1 阶段预测生成 (LID-TCN)，第 2/3 阶段预测细化 (SS-TCN)；
- 损失 = 各阶段加权 CE（类别均衡）+ 截断 MSE 平滑损失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class DualDilatedLayer(nn.Module):
    """双扩展层：并行小/大扩展率分支 + 残差连接。"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 small_dilation=1, large_dilation=8, dropout=0.3):
        super().__init__()
        self.small_conv = nn.Conv1d(
            in_channels, out_channels // 2, kernel_size,
            padding=small_dilation * (kernel_size - 1) // 2,
            dilation=small_dilation)
        self.large_conv = nn.Conv1d(
            in_channels, out_channels // 2, kernel_size,
            padding=large_dilation * (kernel_size - 1) // 2,
            dilation=large_dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res_proj = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):  # x: [B, C_in, T]
        out = torch.cat([self.small_conv(x), self.large_conv(x)], dim=1)
        out = self.dropout(self.relu(self.bn(out)))
        return out + self.res_proj(x)


def _build_tcn_stack(in_channels, hidden, num_layers, dropout):
    """逐层扩展率指数增长的小/大双分支堆叠。"""
    layers = []
    for i in range(num_layers):
        small_d = 2 ** i                    # 1,2,4,... 局部细节
        large_d = 2 ** min(i + 3, 10)       # 8,16,32,... 长程趋势
        layers.append(DualDilatedLayer(
            in_channels if i == 0 else hidden, hidden,
            small_dilation=small_d, large_dilation=large_d, dropout=dropout))
    return nn.ModuleList(layers)


class MSTCNPP(nn.Module):
    """MS-TCN++ for SBR Phase Recognition."""

    def __init__(self, input_dim, num_classes=C.NUM_CLASSES,
                 hidden=C.TCN_HIDDEN, num_stages=C.TCN_NUM_STAGES,
                 stage1_layers=C.TCN_STAGE1_LAYERS,
                 refine_layers=C.TCN_REFINE_LAYERS, dropout=C.TCN_DROPOUT):
        super().__init__()
        self.num_stages = num_stages

        # 第 1 阶段：预测生成 (LID-TCN)
        self.stage1 = _build_tcn_stack(input_dim, hidden, stage1_layers, dropout)
        self.head1 = nn.Conv1d(hidden, num_classes, 1)

        # 第 2..N 阶段：预测细化 (SS-TCN)，输入为上一阶段的类别概率
        self.refine_stages = nn.ModuleList([
            _build_tcn_stack(num_classes, hidden, refine_layers)
            for _ in range(num_stages - 1)])
        self.refine_heads = nn.ModuleList([
            nn.Conv1d(hidden, num_classes, 1) for _ in range(num_stages - 1)])

    def forward(self, x):
        """x: [B, T, F] -> list of [B, num_classes, T]（每阶段一个预测）"""
        x = x.permute(0, 2, 1)  # Conv1d 需要 [B, F, T]

        h = x
        for layer in self.stage1:
            h = layer(h)
        preds = [self.head1(h)]

        for stage, head in zip(self.refine_stages, self.refine_heads):
            h = F.softmax(preds[-1], dim=1)
            for layer in stage:
                h = layer(h)
            preds.append(head(h))
        return preds


class SBRPhaseLoss(nn.Module):
    """多阶段加权交叉熵 + 截断 MSE 平滑损失。"""

    def __init__(self, class_weights=None, lambda_smooth=C.LAMBDA_SMOOTH):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.lambda_smooth = lambda_smooth

    def forward(self, predictions, target):
        """predictions: list of [B,C,T]; target: [B,T]"""
        total = 0.0
        for pred in predictions:
            B, Cc, T = pred.shape
            ce = F.cross_entropy(pred.permute(0, 2, 1).reshape(-1, Cc),
                                 target.reshape(-1),
                                 weight=self.class_weights)
            p = F.softmax(pred, dim=1)
            diff = p[:, :, 1:] - p[:, :, :-1].detach()
            t_mse = torch.mean(torch.clamp(diff ** 2, max=1.0))
            total = total + ce + self.lambda_smooth * t_mse
        return total
