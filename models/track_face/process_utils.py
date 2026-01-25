#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Non-contact-human-physiological-monitoring 
@File    ：process_utils.py
@IDE     ：PyCharm 
@Author  ：王洪飞
@Date    ：2025/12/25 11:10 
'''
from typing import Tuple

import cv2
import numpy as np


def yolo_to_xywh(boxes: np.ndarray, image_size: int = 640) -> np.ndarray:
    """
    将YOLO格式(cx, cy, w, h)直接转换为xywh格式
    Args:
        boxes: [N, 4] 包含 cx, cy, w, h
        image_size: 图像尺寸，用于裁剪
    Returns:
        xywh_boxes: [N, 4] 包含 x, y, w, h
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # 从cx,cy,w,h转换为x,y,w,h
    x = boxes[:, 0] - boxes[:, 2] / 2  # x = cx - w/2
    y = boxes[:, 1] - boxes[:, 3] / 2  # y = cy - h/2
    w = boxes[:, 2]  # 宽度保持不变
    h = boxes[:, 3]  # 高度保持不变

    # 确保不超出边界
    x = np.clip(x, 0, image_size)
    y = np.clip(y, 0, image_size)
    w = np.clip(w, 0, image_size - x)
    h = np.clip(h, 0, image_size - y)

    return np.stack([x, y, w, h], axis=1)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    将xywh格式转换为xyxy格式
    Args:
        boxes: [N, 4] 包含 x, y, w, h
    Returns:
        xyxy_boxes: [N, 4] 包含 x1, y1, x2, y2
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    return np.stack([x1, y1, x2, y2], axis=1)


def process_onnx_output_fast(results: np.ndarray,
                             confidence_th: float = 0.5,
                             iou_threshold: float = 0.5) -> tuple:
    """
    快速处理ONNX模型输出，使用OpenCV NMS
    Args:
        results: ONNX模型输出，形状为 [1, 5, 8400]
        confidence_th: 置信度阈值
        iou_threshold: IoU阈值
    Returns:
        xywh_boxes: 过滤后的框 [M, 4] (xywh格式，相对于640x640)
        scores: 对应的置信度 [M]
    """
    if results is None or results.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

    # 1. 重塑输出
    pred = results[0].T  # 转置为 [8400, 5]

    # 2. 提取坐标和置信度
    boxes_yolo = pred[:, :4]  # [N, 4] 格式: cx, cy, w, h
    scores = pred[:, 4]  # [N] 置信度

    # 3. 置信度过滤
    conf_mask = scores > confidence_th
    boxes_yolo = boxes_yolo[conf_mask]
    scores = scores[conf_mask]

    if len(boxes_yolo) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

    # 4. 转换为xywh格式（用于OpenCV NMS和DeepSort）
    xywh_boxes = yolo_to_xywh(boxes_yolo, image_size=640)

    # 5. 使用OpenCV NMS（需要xywh格式）
    boxes_float = xywh_boxes.astype(np.float32)
    scores_float = scores.astype(np.float32)

    try:
        # 使用OpenCV的NMSBoxes
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_float.tolist(),
            scores=scores_float.tolist(),
            score_threshold=0.0,  # 我们已经过滤了置信度
            nms_threshold=iou_threshold
        )

        if indices is None or len(indices) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

        # 转换为numpy数组
        if hasattr(indices, 'shape'):
            indices = indices.flatten()
        else:
            indices = np.array(indices).flatten()

    except Exception as e:
        print(f"OpenCV NMS error: {e}, using custom NMS")
        # 如果OpenCV NMS失败，使用自定义NMS
        indices = custom_nms(xywh_boxes, scores, iou_threshold)

    # 6. 返回过滤后的结果
    filtered_boxes = xywh_boxes[indices]
    filtered_scores = scores[indices]

    return filtered_boxes, filtered_scores


def custom_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    自定义NMS实现（备用方案）
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    # 从xywh转换为xyxy进行计算
    xyxy_boxes = xywh_to_xyxy(boxes)

    # 按分数降序排序
    order = np.argsort(scores)[::-1]
    xyxy_boxes = xyxy_boxes[order]
    scores = scores[order]

    # 计算所有框的面积
    areas = (xyxy_boxes[:, 2] - xyxy_boxes[:, 0]) * (xyxy_boxes[:, 3] - xyxy_boxes[:, 1])

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # 计算当前框与其他所有框的IoU
        xx1 = np.maximum(xyxy_boxes[i, 0], xyxy_boxes[order[1:], 0])
        yy1 = np.maximum(xyxy_boxes[i, 1], xyxy_boxes[order[1:], 1])
        xx2 = np.minimum(xyxy_boxes[i, 2], xyxy_boxes[order[1:], 2])
        yy2 = np.minimum(xyxy_boxes[i, 3], xyxy_boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-7)

        # 保留IoU低于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def scale_coords_letterbox(boxes: np.ndarray,
                           original_shape: Tuple[int, int],
                           scale: float,
                           left: int,
                           top: int) -> np.ndarray:
    """
    将letterbox处理后的坐标映射回原始图像
    Args:
        boxes: 检测框 [N, 4] (xywh格式，相对于letterbox后的图像)
        original_shape: 原始图像尺寸 (height, width)
        scale: 缩放比例
        left, top: 填充的左上角偏移
    Returns:
        scaled_boxes: 映射回原始图像的框 [N, 4] (xywh格式)
    """
    if len(boxes) == 0:
        return boxes

    h, w = original_shape

    # 去除填充
    boxes[:, 0] = boxes[:, 0] - left
    boxes[:, 1] = boxes[:, 1] - top

    # 缩放回原始尺寸
    boxes[:, 0] = boxes[:, 0] / scale
    boxes[:, 1] = boxes[:, 1] / scale
    boxes[:, 2] = boxes[:, 2] / scale
    boxes[:, 3] = boxes[:, 3] / scale

    # 确保坐标在图像范围内
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - boxes[:, 0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - boxes[:, 1])

    return boxes