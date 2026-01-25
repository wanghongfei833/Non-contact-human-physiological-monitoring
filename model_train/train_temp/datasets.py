#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：main.py 
@File    ：datasets.py
@IDE     ：PyCharm 
@Author  ：王洪飞
@Date    ：2026/1/25 13:13 
'''
import glob
import os
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def letterbox_resize(img, new_size=(640, 640), color=0,pad_rst = False):
    """
    简单的 letterbox resize 函数，支持 float 类型

    Args:
        img: 输入图像 (支持 uint8/float32/float64)
        new_size: 目标尺寸 (width, height)
        color: 填充颜色
        keep_ratio: 是否保持宽高比

    Returns:
        resized_img: 调整后的图像
    """
    h, w = img.shape[:2]
    new_w, new_h = new_size

    # 计算缩放比例
    scale = min(new_w / w, new_h / h)
    nw, nh = int(w * scale), int(h * scale)

    # 选择插值方法
    interpolation = cv2.INTER_LINEAR_EXACT if img.dtype != np.uint8 else cv2.INTER_LINEAR

    # resize
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
    new_img = np.full((new_h, new_w), color, dtype=img.dtype)

    # 将resize后的图像放到中心
    top = (new_h - nh) // 2
    left = (new_w - nw) // 2
    new_img[top:top + nh, left:left + nw] = resized
    if pad_rst:
        return new_img, scale, left, top
    return new_img


class FaceData(Dataset):
    def __init__(self,dir_list:List[str],true_temp:List[float]):
        """
        @param: dir_list 路径的列表
        @param: true_temp 每个路径下对应的真实人脸温度
        """
        super().__init__()
        img_list = []
        true_list = []
        for t,_dirs in zip(true_temp,dir_list):
            sample_data = glob.glob(fr"{_dirs}/*.npy")
            print(f"{_dirs}找到{len(sample_data)}npy文件")
            img_list += sample_data
            true_list += [t for _ in sample_data]
        real_data = [0 < float(os.path.basename(i).strip(".npy").split("-")[0]) < 22 for i in img_list]
        self.img_list = [i for i,j in zip(img_list,real_data) if j]
        self.true_list = [i for i,j in zip(true_list,real_data) if j]
        self.image_mean = 12.131086063726569
        self.image_std = 12.987887632220527
        self.max_temp = 42.0
        self.min_temp = 35.8

    def __len__(self):
        return len(self.img_list)
        # return len(self.input_data)

    def __getitem__(self, item):
        npy_path = self.img_list[item]
        true_temp = self.true_list[item]
        name = os.path.basename(npy_path)
        strips = name.strip(".npy").replace("--", "-").split("-")[:4]
        deep, stand, stand_, bias = map(float, strips)
        img = np.load(npy_path)
        lable = true_temp - img.max()
        img = letterbox_resize(img, (64, 128))
        # 归一化
        img = (img - self.image_mean) / self.image_std
        img = torch.from_numpy(img).float()
        vectors = torch.tensor([deep, stand, stand_, bias])
        return img[None], vectors, torch.tensor(lable).float()

