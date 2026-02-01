#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：main.py 
@File    ：export_model.py
@IDE     ：PyCharm 
@Author  ：王洪飞
@Date    ：2026/1/25 14:11 
'''
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.onnx
import numpy as np

# 获取当前文件的绝对路径
FILE = Path(__file__).resolve()
# 获取项目根目录（假设脚本在项目子目录中）
ROOT = FILE.parents[0]  # 项目根目录
# 添加项目根目录到 Python 路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# 设置工作目录为项目根目录
os.chdir(ROOT)

# 导入模型
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.temp_biase.temp_bias import CrossAttentionRegressionModel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='导出温度偏差校准模型')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--out_dir', type=str, default='exported_models', help='输出目录')
    parser.add_argument('--model_dims', type=int, default=128, help='模型隐藏层维度，默认: 128')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[1, 1, 64, 128],
                        help='输入形状，格式: batch channel height width，默认: 1 1 64 128')
    parser.add_argument('--feature_dim', type=int, default=4,
                        help='特征维度，默认: 4')
    return parser.parse_args()


def export_model():
    """导出模型主函数"""
    args = parse_args()

    # 检查权重文件
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"权重文件不存在: {args.weights}")

    # 创建模型并加载权重
    model = CrossAttentionRegressionModel(hidden_dim=args.model_dims)
    model.load_state_dict(
        torch.load(args.weights, weights_only=True, map_location=torch.device('cpu'))
    )
    model.eval()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 生成输出路径
    model_name = os.path.basename(args.weights).replace('.pth', '.onnx')
    output_file = os.path.join(args.out_dir, model_name)

    # 创建虚拟输入（使用参数或默认值）
    if len(args.input_shape) == 4:
        dummy_image = torch.randn(*args.input_shape)
    else:
        # 如果输入形状不符合要求，使用默认值
        print(f"⚠️  输入形状 {args.input_shape} 不符合要求，使用默认值 [1, 1, 64, 128]")
        dummy_image = torch.randn(1, 1, 64, 128)

    dummy_features = torch.randn(args.input_shape[0], args.feature_dim)

    # 导出为ONNX
    torch.onnx.export(
        model=model,
        args=(dummy_image, dummy_features),
        f=str(output_file),
        input_names=["image_input", "feature_input"],
        output_names=["temperature_output"],
        opset_version=13,
        do_constant_folding=True,
    )

    print(f"✅ 模型导出成功: {output_file}")
    print(f"   图像输入形状: {dummy_image.shape}")
    print(f"   特征输入形状: {dummy_features.shape}")
    print(f"   模型维度: {args.model_dims}")

    return output_file


if __name__ == "__main__":
    export_model()