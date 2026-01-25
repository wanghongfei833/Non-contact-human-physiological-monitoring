#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Non-contact-human-physiological-monitoring 
@File    ：base_utils.py
@IDE     ：PyCharm 
@Author  ：王洪飞
@Date    ：2025/12/25 10:56 
'''
import os
import sys

import cv2
import numpy as np
import logging

import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
from typing import Tuple, Optional, Union


def letterbox_resize(
        img: np.ndarray,
        new_size: Tuple[int, int] = (640, 640),
        color: Union[int, float, Tuple[int, int, int]] = 0,
        pad_rst: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float, int, int]]:
    """
    简单的 letterbox resize 函数，支持 uint8/float32/float64 类型

    Args:
        img: 输入图像 (H, W) 或 (H, W, 3)
        new_size: 目标尺寸 (width, height)
        color: 填充颜色
            - 对于灰度图: 可以是标量 (int/float) 或单元素元组
            - 对于彩色图: 必须是三元素元组 (B, G, R)
        pad_rst: 是否返回填充信息

    Returns:
        如果 pad_rst=False: 调整后的图像
        如果 pad_rst=True: (调整后的图像, 缩放比例, 左填充, 上填充)
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be numpy array, got {type(img)}")

    if img.ndim not in (2, 3):
        raise ValueError(f"img must have 2 or 3 dimensions, got {img.ndim}")

    h, w = img.shape[:2]
    new_w, new_h = new_size

    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"new_size must be positive, got {new_size}")

    # 计算缩放比例
    scale = min(new_w / w, new_h / h)
    nw, nh = int(w * scale), int(h * scale)

    # 确保尺寸不为0
    nw = max(1, nw)
    nh = max(1, nh)

    # 选择插值方法
    if img.dtype == np.uint8:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_LINEAR_EXACT

    # 调整大小
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)

    # 处理颜色参数
    if img.ndim == 2:
        # 灰度图
        if isinstance(color, (tuple, list)):
            fill_value = color[0] if len(color) > 0 else 0
        else:
            fill_value = color
        new_img = np.full((new_h, new_w), fill_value, dtype=img.dtype)

    elif img.ndim == 3:
        # 彩色图
        if img.shape[2] != 3:
            raise ValueError(f"Color image must have 3 channels, got {img.shape[2]}")

        if isinstance(color, (int, float)):
            # 如果是标量，扩展为三个通道
            color_tuple = (color, color, color)
        elif isinstance(color, (tuple, list)):
            if len(color) != 3:
                raise ValueError(f"Color must be a 3-element tuple for color images, got {color}")
            color_tuple = tuple(color)
        else:
            raise TypeError(f"Color must be int, float, or tuple/list, got {type(color)}")

        new_img = np.full((new_h, new_w, 3), color_tuple, dtype=img.dtype)

    # 将resize后的图像放到中心
    top = (new_h - nh) // 2
    left = (new_w - nw) // 2

    # 将调整后的图像放入新图像中心
    if img.ndim == 2:
        new_img[top:top + nh, left:left + nw] = resized
    else:
        new_img[top:top + nh, left:left + nw, :] = resized

    if pad_rst:
        return new_img, scale, left, top
    return new_img
# 设置日志
def setup_logging(log_file):


    # 配置根日志记录器
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 设置特定模块的日志级别
    logging.getLogger('PySide6').setLevel(logging.WARNING)
    logging.getLogger('onnxruntime').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {os.path.abspath(log_file)}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")

    return logger


def is_wsl():
    """检测是否是 WSL 环境"""
    system = platform.system().lower()
    if system != 'linux':
        return False

    # 多种检测方法
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                return True
    except:
        pass

    if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
        return True

    if 'WSL_DISTRO_NAME' in os.environ:
        return True

    return False
def open_folder(path):
    """修复版本：支持 WSL"""
    folder_path = Path(path)

    if not folder_path.exists() or not folder_path.is_dir():
        return False

    abs_path = folder_path.absolute()

    # 检测 WSL
    if is_wsl():
        print("检测到 WSL 环境，使用 Windows 资源管理器")

        # 将 WSL 路径转换为 Windows 路径
        wsl_path = str(abs_path)

        # 使用 wslpath 工具转换（WSL 自带）
        try:
            result = subprocess.run(
                ['wslpath', '-w', wsl_path],
                capture_output=True,
                text=True,
                check=True
            )
            windows_path = result.stdout.strip()
        except Exception as e:
            print(f"wslpath 转换失败: {e}")
            # 尝试手动转换（仅处理 /mnt/ 开头的路径）
            if wsl_path.startswith('/mnt/'):
                parts = wsl_path.split('/')
                if len(parts) >= 3:
                    drive = parts[2].upper()
                    rest = '\\'.join(parts[3:])
                    windows_path = f"{drive}:\\{rest}"
                else:
                    windows_path = wsl_path
            else:
                print("无法转换非 /mnt/ 路径")
                return False

        # 使用 Windows 的 explorer.exe
        cmd = ['cmd.exe', '/c', 'start', 'explorer', windows_path]
        subprocess.Popen(cmd, shell=False)

    else:
        # 原来的代码
        system = platform.system().lower()

        if system == 'windows':
            subprocess.Popen(f'explorer "{abs_path}"', shell=True)

        elif system == 'darwin':
            subprocess.Popen(['open', str(abs_path)])

        elif system == 'linux':
            file_managers = [
                ['xdg-open', str(abs_path)],
                ['nautilus', str(abs_path)],
                ['dolphin', str(abs_path)],
                ['thunar', str(abs_path)],
                ['pcmanfm', str(abs_path)],
                ['nemo', str(abs_path)],
            ]

            for cmd in file_managers:
                try:
                    subprocess.Popen(cmd)
                    break
                except FileNotFoundError:
                    continue
            else:
                print("没有找到可用的文件管理器")
                return False
        else:
            print(f"不支持的操作系统: {system}")
            return False

    return True