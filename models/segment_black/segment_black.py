import time

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50


class SegmentBlack(object):
    def __init__(self, deep_path="./weights/segmentation_model.pth"):
        self.model = self.initialize_model(2, torch.device("cuda")).half()
        self.load_state(deep_path)
        self.model.eval()
        self.kernel = np.ones(3, np.uint8)

    def load_state(self, weight):
        self.model.load_state_dict(torch.load(weight, weights_only=True))

    @staticmethod
    def initialize_model(num_classes, device) -> torch.nn.Module:
        model = deeplabv3_resnet50(pretrained=True)
        # 修改分类器以适应我们的类别数
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        return model.to(device)

    @staticmethod
    def process_rgb(rgb_frame: np.ndarray):
        h, w = rgb_frame.shape[:2]
        rgb_frame = cv2.resize(rgb_frame, (512, 512))
        rgb_frame = np.transpose(rgb_frame / 255., (2, 0, 1))
        rgb_tensor = torch.from_numpy(rgb_frame).to(torch.float16)
        return rgb_tensor[None,], (h, w)

    def infer(self, rgb_frame):
        tensor_frame, hw = self.process_rgb(rgb_frame)
        tensor_frame = tensor_frame.cuda()
        with torch.no_grad():
            out = torch.argmax(self.model(tensor_frame)['out'], dim=1)[0]  # mask
            out = out.cpu().numpy().astype(np.uint8)
        out = cv2.resize(out, (hw[1], hw[0]))

        # 第二次腐蚀
        x, y, w, h = self.get_largest_contour_bbox(out)  # 最大面积的 bbox
        if x is None:
            # return None, None
            return None,[387,693,462,771]
        else:
            return -1, [x, y, x + w, y + h]

    @staticmethod
    def get_largest_contour_bbox(binary_image):
        """
        获取二值图像中最大轮廓的边框

        参数:
            binary_image: 二值图像（单通道，0=背景，255=前景）

        返回:
            bbox: 最大轮廓的边框 (x, y, w, h)
            contour: 最大轮廓的点集
        """
        # 确保输入是二值图像
        if len(binary_image.shape) > 2:
            raise ValueError("输入图像应该是单通道二值图像")

        # 查找所有轮廓
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,  # 只检测外部轮廓
            cv2.CHAIN_APPROX_SIMPLE  # 简化轮廓点
        )

        # 如果没有找到轮廓
        if not contours:
            return None, None, None, None

        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 获取轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(largest_contour)

        return x, y, w, h
