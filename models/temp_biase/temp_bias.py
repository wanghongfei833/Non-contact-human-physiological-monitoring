import torch
from torch import nn
import cv2
import numpy as np
import onnxruntime as ort
from utils.base_utils import *


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


class TempBias(object):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        if model_path.endswith(".onnx"):
            self.infer_model = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider','CPUExecutionProvider'],

            )
            inputs = self.infer_model.get_inputs()
            outputs = self.infer_model.get_outputs()
            self.input_name1 = inputs[0].name  # 应该是 'input'
            self.input_name2 = inputs[1].name  # 应该是 'onnx::Gemm_1'
            self.output_name = outputs[0].name  # 获取输出名称

            # 准备测试数据，必须按照模型期望的形状
            input_data_1 = np.random.random((1, 1, 128, 64)).astype(np.float32)  # 第一个输入
            input_data_2 = np.random.random((1, 4)).astype(np.float32)  # 第二个输入

            # 正确的输入方式：使用字典，键为输入名称
            input_feed = {
                self.input_name1: input_data_1,
                self.input_name2: input_data_2
            }

            # 运行推理
            self.infer_model.run(None, input_feed)
            print("TempBias ONNX model is ready...")


        elif model_path.endswith(".pt") or model_path.endswith(".pth"):
            self.infer_model = CrossAttentionRegressionModel(hidden_dim=128).cuda()#训练的新模型，模型大小不一致
            self.infer_model.load_state_dict(torch.load(model_path, weights_only=True))
            self.infer_model.eval()
            torch.onnx.export(
                model=self.infer_model,
                args=(torch.randn(1, 1, 64, 128).cuda(),torch.randn(1, 4).cuda()),
                f=model_path.replace(".pth", ".onnx"),
                input_names=["input"],  # 输入名称
                output_names=["output"],  # 输出名称
                dynamic_axes=None,  # 固定尺寸
                # opset_version=12,  # ONNX算子集版本
                do_constant_folding=True,  # 常量折叠优化
                # verbose=True  # 显示详细信息
            )
            print("torch running")
        self.image_mean = 12.131086063726569
        self.image_std = 12.987887632220527

    def infers_torch(self, image, deep, stand, stand_, bias):
        # 图像初始化
        # img = letterbox_resize(image, (64, 128))
        img = letterbox_resize(image, (64, 128))
        # 归一化
        img = (img - self.image_mean) / self.image_std
        img = torch.from_numpy(img).float()[None, None]
        vector = torch.tensor([deep, stand, stand_, bias]).float()
        vector = vector.resize(1, 4)
        with torch.no_grad():
            img = img.cuda()
            vector = vector.cuda()
            bias_model = self.infer_model(img, vector).cpu()
        return bias_model
    def infers_batch(self,image,deep, vector):
        # temp_bias_model.infers(face_temp_arr, depth, standard_black, standard_black_, temp_bias)
        img = [letterbox_resize(i,(64,128)) for i in image]
        img = [ (i - self.image_mean) / self.image_std for i in img]
        img = np.array(img).astype(np.float32)[:,np.newaxis]
        deep = np.array(deep).reshape(-1,1)
        vector_ = np.tile(vector,(deep.shape[0],1))
        vector_data = np.concatenate([deep, vector_], axis=1,dtype=np.float32)
        input_feed = {
            self.input_name1: img,
            self.input_name2: vector_data
        }
        # 运行推理
        bias_model = self.infer_model.run([self.output_name], input_feed)[0]    # 第0个输出，只有一个输出为 输出形状 B N
        # 消除最后维度
        bias_model = bias_model.reshape(-1)
        return bias_model.tolist()