#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：main.py 
@File    ：export_onnx.py
@IDE     ：PyCharm 
@Author  ：王洪飞
@Date    ：2026/1/25 15:14 
'''
import os
import sys
from pathlib import Path

import torch
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
from EfficientPhys import EfficientPhys

# 导出
pth_path = r"./runs/exp/PreTrainedModels/PURE_PURE_UBFC_efficientphys_Epoch79.pth"
onnx_path = r"./runs/exp/PreTrainedModels/PURE_Efficientphys_Epoch79_10x3x72x72.onnx"


# onnx_path = r"./UBFC-rPPG_EfficientPhys.onnx"

def export_efficientphys_to_onnx():
    """导出EfficientPhys模型到ONNX格式"""

    # 1. 加载模型
    model = EfficientPhys(frame_depth=10, img_size=72)

    # 2. 加载权重
    checkpoint_path = pth_path
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # 3. 处理状态字典（解决可能的键名不匹配）
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除可能的"module."前缀（如果是多GPU训练保存的）
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    # 4. 加载权重
    model.load_state_dict(new_state_dict)
    model.eval()

    # 5. 准备示例输入 - 重要：需要模拟训练时的预处理
    # 训练时输入形状: (N, D, C, H, W) = (batch_size, 10, 3, 72, 72)
    # 在forward之前，训练代码做了以下处理：
    # 1) data.view(N*D, C, H, W) -> (N*D, C, H, W)
    # 2) 截断到base_len的整数倍
    # 3) 添加额外一帧 -> 变为 (N*D+1, C, H, W)
    # 4) 在forward中会reshape回 (N, D+1, C, H, W)

    # 所以对于单个样本，模型实际处理的是 11 帧 (10+1)
    batch_size = 1
    frame_depth = 10
    channels = 3
    height = 72
    width = 72

    # 创建示例输入
    dummy_input = torch.randn(batch_size * frame_depth + 1, channels, height, width)

    # 6. 模拟训练时的预处理步骤
    with torch.no_grad():
        # 获取模型在给定输入下的输出
        # 注意：EfficientPhys的forward方法应该会处理这些预处理
        dummy_output = model(dummy_input)
        print(f"模型输出形状: {dummy_output.shape}")

        # 获取实际的处理后输入（用于导出时验证）
        # 这需要查看EfficientPhys的forward方法
        # 如果forward方法内部有预处理，我们需要确保导出时能正确处理

    # 7. 导出ONNX模型
    # onnx_path = r"./runs/exp/PreTrainedModels/PURE_PURE_UBFC_efficientphys_Epoch29.onnx"

    # 设置动态轴
    dynamic_axes = {
        'input': {0: 'batch_size'},  # 批次维度动态
        'output': {0: 'batch_size'}  # 输出批次维度动态
    }

    # 导出选项
    export_params = True
    opset_version = 11
    do_constant_folding = True
    input_names = ['input']
    output_names = ['output']

    # 8. 导出模型
    torch.onnx.export(
        model,  # 要导出的模型
        dummy_input,  # 模型输入（或元组）
        onnx_path,  # 输出文件路径
        export_params=export_params,  # 存储训练好的参数
        opset_version=opset_version,  # ONNX版本
        do_constant_folding=do_constant_folding,  # 优化常量
        input_names=input_names,  # 输入名称
        output_names=output_names,  # 输出名称
        dynamic_axes=dynamic_axes,  # 动态轴
        verbose=False  # 是否打印详细信息
    )

    print(f"模型已成功导出到: {onnx_path}")

    # 9. 验证导出的模型
    verify_onnx_model(onnx_path, dummy_input.numpy(), dummy_output.numpy())

    return onnx_path


def verify_onnx_model(onnx_path, test_input, expected_output, rtol=1e-3, atol=1e-5):
    """验证导出的ONNX模型"""
    import onnx
    import onnxruntime as ort

    print("\n" + "=" * 50)
    print("验证ONNX模型")
    print("=" * 50)

    # 1. 检查模型结构
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX模型结构验证通过")

    # 2. 使用ONNX Runtime运行推理
    ort_session = ort.InferenceSession(onnx_path)

    # 获取输入输出名称
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # 运行推理
    ort_inputs = {input_name: test_input}
    ort_outputs = ort_session.run([output_name], ort_inputs)

    # 3. 比较输出
    ort_output = ort_outputs[0]

    print(f"原始PyTorch输出形状: {expected_output.shape}")
    print(f"ONNX Runtime输出形状: {ort_output.shape}")

    # 计算差异
    diff = np.abs(expected_output - ort_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")

    # 检查是否在容差范围内
    if np.allclose(expected_output, ort_output, rtol=rtol, atol=atol):
        print("✓ ONNX模型输出与PyTorch模型匹配")
    else:
        print("✗ ONNX模型输出与PyTorch模型不匹配")

    return ort_output


def export_with_custom_forward():
    """如果EfficientPhys的forward有特殊处理，可能需要自定义"""

    class EfficientPhysWrapper(torch.nn.Module):
        """包装EfficientPhys，模拟训练时的预处理"""

        def __init__(self, efficientphys_model, base_len=10, num_gpu=1):
            super().__init__()
            self.model = efficientphys_model
            self.base_len = base_len
            self.num_gpu = num_gpu

        def forward(self, data):
            """模拟训练时的预处理步骤"""
            # 获取输入形状
            N, D, C, H, W = data.shape

            # 1. reshape: (N, D, C, H, W) -> (N*D, C, H, W)
            data = data.view(N * D, C, H, W)

            # 2. 截断到base_len的整数倍
            data = data[:(N * D) // self.base_len * self.base_len]

            # 3. 为EfficientPhys添加额外帧（因为使用torch.diff）
            last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_gpu, 1, 1, 1)
            data = torch.cat((data, last_frame), 0)

            # 4. 现在输入形状为 (N*D+1, C, H, W)
            # 在EfficientPhys的forward中，这会被reshape为 (N, D+1, C, H, W)

            # 5. 调用原始模型
            output = self.model._forward_impl(data)  # 假设模型有这个内部方法
            return output

    # 加载模型
    original_model = EfficientPhys(frame_depth=10, img_size=72)
    checkpoint = torch.load(r"./runs/exp/PreTrainedModels/PURE_PURE_UBFC_efficientphys_Epoch29.pth")

    # 处理状态字典
    state_dict = checkpoint
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # 加载权重
    original_model.load_state_dict(state_dict, strict=False)
    original_model.eval()

    # 创建包装模型
    wrapped_model = EfficientPhysWrapper(original_model, base_len=10, num_gpu=1)
    wrapped_model.eval()

    # 准备输入
    dummy_input = torch.randn(10, 3, 72, 72)

    # 导出
    # onnx_path = r"./runs/exp/PreTrainedModels/PURE_PURE_UBFC_efficientphys_Epoch29_wrapped.onnx"

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"包装模型已导出到: {onnx_path}")
    return onnx_path


def export_simple_version():
    """简化版导出，假设模型forward已经处理了所有预处理"""

    # 加载模型
    model = EfficientPhys(frame_depth=10, img_size=72)

    # 加载权重
    checkpoint = torch.load(
        # r"./runs/exp/PreTrainedModels/PURE_PURE_UBFC_efficientphys_Epoch29.pth",
        r"./final_model_release\PURE_EfficientPhys.pth",
        map_location='cpu'
    )

    # 处理状态字典
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 移除可能的"module."前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    # 加载权重
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 创建示例输入
    # 注意：根据训练代码，模型应该接收 (batch_size, frame_depth, C, H, W)
    # 但在forward内部，它会处理成 (batch_size, frame_depth+1, C, H, W)
    dummy_input = torch.randn(11, 3, 72, 72)  # 使用batch_size=2测试

    # 测试模型
    with torch.no_grad():
        output = model(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # 批次维度动态
            'output': {0: 'batch_size'}
        },
        verbose=True
    )

    # 验证
    verify_onnx_model_simple(onnx_path, dummy_input.numpy(), output.numpy())

    return onnx_path


def verify_onnx_model_simple(onnx_path, test_input, expected_output):
    """简化验证"""
    import onnxruntime as ort

    ort_session = ort.InferenceSession(onnx_path)

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    ort_output = ort_session.run([output_name], {input_name: test_input})[0]

    print(f"\n验证结果:")
    print(f"PyTorch输出形状: {expected_output.shape}")
    print(f"ONNX输出形状: {ort_output.shape}")

    # 计算相对误差
    relative_error = np.abs(expected_output - ort_output) / (np.abs(expected_output) + 1e-8)
    max_relative_error = np.max(relative_error)

    print(f"最大相对误差: {max_relative_error:.6f}")

    if max_relative_error < 1e-3:
        print("✓ 模型导出成功，误差在可接受范围内")
    else:
        print("⚠ 模型导出可能存在较大误差")

    return ort_output


# 主程序
if __name__ == "__main__":
    print("开始导出EfficientPhys模型到ONNX格式")
    print("-" * 50)

    # 方法1: 简单导出
    print("\n方法1: 简单导出")
    try:
        onnx_path = export_simple_version()
        print(f"模型已导出到: {onnx_path}")
    except Exception as e:
        print(f"简单导出失败: {e}")

    # 方法2: 如果需要特殊处理，使用包装模型
    # print("\n方法2: 使用包装模型导出")
    # try:
    #     onnx_path = export_with_custom_forward()
    #     print(f"包装模型已导出到: {onnx_path}")
    # except Exception as e:
    #     print(f"包装模型导出失败: {e}")

    # 方法3: 完整导出
    print("\n方法3: 完整导出")
    try:
        onnx_path = export_efficientphys_to_onnx()
    except Exception as e:
        print(f"完整导出失败: {e}")
        import traceback

        traceback.print_exc()