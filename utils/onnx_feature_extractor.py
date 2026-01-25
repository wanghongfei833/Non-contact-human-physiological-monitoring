import onnxruntime as ort
import numpy as np
import cv2
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ONNX特征提取器类
class ONNXFeatureExtractor:
    """
    使用ONNX模型运行DeepSORT特征提取的类
    完全替代原来的MobileNetv2_Embedder
    """

    def __init__(
            self,
            onnx_model_path: str,
            max_batch_size: int = 16,
            bgr: bool = True,
            use_gpu: bool = True,
            use_fp16: bool = False
    ):
        """
        初始化ONNX特征提取器
        """
        self.max_batch_size = max_batch_size
        self.bgr = bgr
        self.use_fp16 = use_fp16

        # 检查GPU可用性
        available_providers = ort.get_available_providers()
        print(f"可用的ONNX Runtime提供者: {available_providers}")

        # 设置执行提供者（优先使用CPU，避免CUDA问题）
        providers = []
        if use_gpu and 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            print("使用GPU加速")
        else:
            print("使用CPU执行（GPU不可用或已禁用）")

        # 总是添加CPU作为备用
        if 'CPUExecutionProvider' not in providers:
            providers.append('CPUExecutionProvider')

        # 加载ONNX模型
        print(f"加载ONNX特征提取模型: {onnx_model_path}")
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=providers
        )

        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # 通常为 [batch, 3, 224, 224]

        # 解析输入尺寸
        _, _, self.input_height, self.input_width = self.input_shape
        print(f"模型输入尺寸: {self.input_width}x{self.input_height}")

        # 预热模型
        self._warmup()

        print(f"ONNX特征提取器初始化完成")

    def _warmup(self):
        """预热模型"""
        dummy_input = np.random.randn(1, 3, self.input_height, self.input_width)
        dummy_input = dummy_input.astype(np.float32)  # 确保是float32

        self.session.run(None, {self.input_name: dummy_input})
        print("模型预热完成")

    def preprocess(self, np_image: np.ndarray) -> np.ndarray:
        """
        预处理单张图像，与原始MobileNetv2_Embedder.preprocess完全一致
        返回float32类型
        """
        # 1. BGR转RGB（如果需要）
        if self.bgr:
            np_image_rgb = np_image[..., ::-1].copy()
        else:
            np_image_rgb = np_image.copy()

        # 2. 调整大小到模型输入尺寸
        input_image = cv2.resize(np_image_rgb, (self.input_width, self.input_height))

        # 3. 转换为float32并归一化到[0, 1]
        input_image = input_image.astype(np.float32) / 255.0

        # 4. 通道顺序调整为 (C, H, W)
        input_image = input_image.transpose(2, 0, 1)

        # 5. ImageNet归一化（使用float32）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        input_image = (input_image - mean) / std

        # 6. 添加批次维度
        input_image = np.expand_dims(input_image, axis=0)

        return input_image

    def _batch_generator(self, items, batch_size):
        """批次生成器"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def predict(self, np_images: list) -> list:
        """
        批量推理，确保输入为float32
        """
        if not np_images:
            return []

        all_features = []

        # 批量预处理
        preprocessed_images = []
        for img in np_images:
            preprocessed = self.preprocess(img)
            preprocessed_images.append(preprocessed)

        # 分批处理
        for batch in self._batch_generator(preprocessed_images, int(self.max_batch_size)):
            # 堆叠批次并确保为float32
            batch_input = np.vstack(batch)

            # 关键：确保是float32，避免double类型
            if batch_input.dtype != np.float32:
                batch_input = batch_input.astype(np.float32)

            # ONNX推理
            try:
                outputs = self.session.run(None, {self.input_name: batch_input})
                features = outputs[0]
                all_features.extend(features)
            except Exception as e:
                print(f"推理失败: {e}")
                # 返回空特征列表
                all_features.extend([np.zeros(1280, dtype=np.float32)] * len(batch))

        return all_features

    def __call__(self, np_images: list) -> list:
        """使类可调用"""
        return self.predict(np_images)