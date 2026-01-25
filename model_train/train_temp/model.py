#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：main.py 
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：王洪飞
@Date    ：2026/1/25 13:13 
'''
import torch
from torch import nn


class CrossAttentionRegressionModel(nn.Module):
    def __init__(self, input_channels=1, num_extra_features=4, hidden_dim=512):
        super(CrossAttentionRegressionModel, self).__init__()

        # 图像特征提取 64 128
        dim_image = 64
        self.img_encoder = nn.Sequential(
            nn.Conv2d(input_channels, dim_image, 7, padding=3, bias=False),
            nn.BatchNorm2d(dim_image),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(dim_image, dim_image * 2, 7, padding=3, bias=False, stride=1),
            nn.BatchNorm2d(dim_image * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(dim_image * 2, dim_image * 4, 7, padding=3, bias=False, stride=1),
            nn.BatchNorm2d(dim_image * 4),
            nn.AdaptiveAvgPool2d((4, 4))  # 保持空间信息
        )

        # 图像特征变换
        self.img_proj = nn.Linear(256 * 16, hidden_dim)

        # 向量特征变换
        self.vec_proj = nn.Sequential(
            nn.Linear(num_extra_features, hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),

        )

        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # 融合后的预测网络
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, img, extra_features):
        batch_size = img.size(0)

        # 提取图像特征
        img_feat = self.img_encoder(img)  # [B, 64, 4, 4]
        img_feat = img_feat.view(batch_size, -1)  # [B, 64 * 4 * 4]
        img_feat = self.img_proj(img_feat)  # [B, hidden_dim]
        img_feat = img_feat.unsqueeze(1)  # [B, 1, hidden_dim]

        # 处理向量特征
        vec_feat = self.vec_proj(extra_features)  # [B, hidden_dim]
        vec_feat = vec_feat.unsqueeze(1)  # [B, 1, hidden_dim]

        # 交叉注意力：图像作为query，向量作为key/value
        attended_img, _ = self.cross_attention(
            query=img_feat,
            key=vec_feat,
            value=vec_feat
        )

        # 另一种交叉注意力：向量作为query，图像作为key/value
        attended_vec, _ = self.cross_attention(
            query=vec_feat,
            key=img_feat,
            value=img_feat
        )

        # 拼接两种注意力结果
        combined = torch.cat([
            attended_img.squeeze(1),
            attended_vec.squeeze(1)
        ], dim=1)  # [B, hidden_dim * 2]

        # 预测
        output = self.predictor(combined)
        return output.squeeze()
