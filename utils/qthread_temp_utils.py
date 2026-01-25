"""
主要功能为 一直计算温度，一旦温度流送入 则计算bbox对应的温度信息

"""
import os
from math import sqrt as m_sqrt
import torch
from torch import nn
from models.temp_biase.temp_bias import TempBias
from utils.camera_stream import *
from utils.log import LOG


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


def letterbox_resize(img, new_size=(640, 640), color=0):
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

    return new_img


class TempModelDepth(object):
    """
    根据深度信息 矫正温度信息
    """

    def __init__(self, TEMP_MODEL_PATH, infer_size=(64, 128)):
        super().__init__()
        self.model = CrossAttentionRegressionModel(hidden_dim=256)
        load_weight = torch.load(TEMP_MODEL_PATH, map_location='cpu', weights_only=True)
        self.model.load_state_dict(load_weight)
        self.model.cuda()
        self.model.eval()
        self.infer_size = infer_size
        self.image_mean = 12.131086063726569
        self.image_std = 12.987887632220527

    def process_data_torch(self, temp_arr: list, depth: list, stand: float, stand_: float, temp_bias: float):
        # 先进行letterbbox
        temp_arr = [letterbox_resize(i, self.infer_size).transpose(2, 0, 1)[None,] for i in temp_arr]
        temp_arr = np.concatenate(temp_arr, axis=0)
        temp_tensor = torch.from_numpy(temp_arr).cuda()
        # 组装 depth 和 stand等信息 按照 vector = torch.tensor([deep, stand, stand_, bias]).float() 这个顺序进行组装
        # 根据 depth是list而 stand是float，需要进行广播
        stand = torch.tensor([stand, stand_, stand_, temp_bias]).float().cuda()[None, :]  # B C
        depth = torch.tensor(depth).cuda()[None, :]  # B 1
        vector = torch.cat((depth, stand), dim=1)
        # 归一化
        temp_tensor = (temp_tensor - self.image_mean) / self.image_std  # B C H W
        return temp_tensor, vector

    def forward_torch(self, temp_tensor, vector):
        return self.model(temp_tensor, vector).cpu().tolist()


def correct_temperature(D10, E5, E3=0.98, E4=1.0):
    """

    :param D10: 目标温度
    :param E5:  距离
    :param E3:  黑体的发射率
    :param E4:  大气透射率
    :return:
    """

    # E3=0.98             # fashelv
    # E4=1                # toushelv
    # D10 = 36.5            # wendu C
    # E5 = 1.7            # juli
    D6 = 25  # huanjingwendu C
    E6 = D6 + 273  # kaierwen
    E7 = E6 ** 4  # huanjingwendu -->E6 ^ 4
    D8 = D6
    E8 = D8 + 273
    E9 = E8 ** 4
    E10 = D10 + 273
    E11 = E10 ** 4

    E12 = m_sqrt(m_sqrt(
        ((E11 - E7) / (E3 * E4) - (E9 - E7) / E3 + E9)
    ))

    data = (E12 + (E5 * 0.85 - 1.125) * (E12 - E6) / 100) * 10 - 2730
    return data / 10



class TemperatureCalibration(QThread):
    temp_signal = Signal(str)
    signal = Signal(list)
    def __init__(self, logger: LOG, config_data):
        super().__init__()
        self.logger = logger
        self.running = True
        # 取出基本配置信息
        BLACK = config_data['BLACK']
        FACE_TEMP = config_data['FACE_TEMP']
        # 温度校准模型
        self.model = TempBias(FACE_TEMP['TEMP_MODEL_PATH'])
        self.logger.log_info_enhanced("温度校准模型加载成功")
        self.stand_temp = BLACK['standard_temp']
        self.stand_temp_ = correct_temperature(self.stand_temp, BLACK['standard_depth'])
        self.temp_bias = self.stand_temp - self.stand_temp_

        self.vector = np.array([self.stand_temp, self.stand_temp_,self.temp_bias]).reshape(1,3).astype(np.float32)
        self.standard_depth = BLACK['standard_depth']
        # 记录温度流的筛选
        self.fpr = FACE_TEMP['FPR']
        self.whs = FACE_TEMP['WHS']
        self.pix_size = FACE_TEMP['PixSize']
        self.temp_arr_thr = FACE_TEMP['TAT']
        self.data_collection = FACE_TEMP['data_collection']
        self.data_collection_path = FACE_TEMP['data_collection_path']
        self.save_info = False # 数据将保存
        if self.data_collection and os.path.exists(self.data_collection_path):
            self.logger.log_info_enhanced("为确保数据不被污染，数据存储路径不能重复。",'ERROR',font_size=20)
            self.logger.log_info_enhanced("文件夹:{}已经存在，请重新修改路径".format(self.data_collection_path),'ERROR',font_size=20)
            self.logger.log_info_enhanced("数据将不会采集",'ERROR',font_size=20)
        if self.data_collection and not os.path.exists(self.data_collection_path):
            os.makedirs(self.data_collection_path)
            self.save_info = True  # 数据将保存
            self.logger.log_info_enhanced("数据将保存至:{}".format(self.data_collection_path))


        # 定义暂停信号
        self.pause = False

    def run(self) -> None:
        while self.running:
            if self.pause:
                time.sleep(0.1)
                continue
            try:
                temp_arr = gray2temp_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # print("temp_arr",temp_arr.shape )
            # temp_arr = cv2.resize(temp_arr, (902, 1177))
            # 获取 人脸检测部分发送过来的 信息
            # {"id": tracker_id, "bbox": bbox, "depth": depth_value, "delete_ids": del_ids}
            # 关键步骤计时
            try:
                # 使用带超时的get，可以检查停止事件
                face_detect_info = face2temp_queue.get(timeout=0.1)
            except queue.Empty:
                # 队列为空，继续循环
                continue

            if len(face_detect_info) != 0:
                ids = face_detect_info['id']
                bbox = face_detect_info['bbox']
                depth = face_detect_info['depth']
                try:
                    ids,  depth, temp_arr_bbox,temp_value_list,bbox = self._check_bbox(temp_arr, bbox, ids, depth)
                    if self.data_collection and self.save_info:
                        for depth_value,box in zip(depth,bbox):
                            txt = f"{depth_value:.4f}-{self.vector[0]:.2f}-{self.vector[1]:.2f}-{self.vector[2]:.2f}"
                            face_temp_arr = temp_arr[box[1]:box[3],box[0]:box[2]]
                            save_name = os.path.join(f"./{self.data_collection_path}/{txt}.npy")
                            normalized_uint8 = cv2.normalize(face_temp_arr, None, 0, 255, cv2.NORM_MINMAX)
                            normalized_uint8 = normalized_uint8.astype(np.uint8)
                            np.save(save_name, face_temp_arr)
                            cv2.imwrite(os.path.join(f"./{self.data_collection_path}/{txt}.jpg"), normalized_uint8)


                except Exception as e:
                    self.logger.log_info_enhanced(f'温度信息处理错误(_check_bbox校准框):{e}',"ERROR")
                    continue
                if len(ids)>=1:
                    try:
                        temp_bias = self.model.infers_batch(temp_arr_bbox,depth,self.vector)
                        if not isinstance(temp_bias, list) and len(temp_value_list)==1:
                            temp_bias = [temp_bias]
                         # temp转为temp_new (结合bias)
                        temp_value_list = [i+j for i,j in zip(temp_value_list, temp_bias)]
                    except Exception as e:
                        self.logger.log_info_enhanced(f'温度信息处理错误(模型推理):{e}',"ERROR")
                        continue
                    try:
                        queue_push(temp2display_queue,{"ids":ids,"temp":temp_value_list,"bbox":bbox,"depth":depth})
                    except Exception as e:
                        self.logger.log_info_enhanced(f'温度信息处理错误(队列推送):{e}',"ERROR")

    def _check_bbox(self, temp_arr, bbox, ids, depth):
        """

        Args:
            temp_arr: 输入的图像
            bbox: 框信息
            ids: 人员ID
            depth: 深度信息

        Returns:

        """
        # ----------- 计算温度信息 --------------------

        # 获取温度 框 同时图像只保留 上半部分

        temp_arr_bbox = [(temp_arr[y1:y1 + (y2 - y1) // 2, x1:x2]) for (x1, y1, x2, y2) in bbox]
        # 去除 人脸占比过少的框
        temp_arr_bbox_ratio = [(i >self.temp_arr_thr).sum() > i.size * self.fpr for i in temp_arr_bbox]

        # 人脸框高宽占比异常
        temp_arr_bbox_scale = [((x2 - x1) / (y2 - y1)) for (x1, y1, x2, y2) in bbox]
        temp_arr_bbox_scale = [(i < self.whs) or (i > 1 / self.whs) for i in temp_arr_bbox_scale]
        # 剔除过于小的检测框 很有可能检测错误
        temp_arr_bbox_pix = [i.size > self.pix_size for i in temp_arr_bbox]

        # 经过上述值筛选后 剩余的框为合理框
        temp_arr_bbox_bools = [temp_arr_bbox_ratio[i] and \
                               temp_arr_bbox_pix[i] and \
                               temp_arr_bbox_scale[i]
                               for i in range(len(temp_arr_bbox))
                               ]
        temp_value_list = [i.max() if i.size > 0 else 0. for i in temp_arr_bbox]

        #  判断全部删除情况
        if not any(temp_arr_bbox_bools):
            return [],[],[],[],[]
        else:
            # 将多余的id、bbox、depth去除
            ids = [ids[i] for i in range(len(ids)) if temp_arr_bbox_bools[i]]
            bbox = [bbox[i] for i in range(len(bbox)) if temp_arr_bbox_bools[i]]
            temp_arr_bbox = [temp_arr_bbox[i] for i in range(len(temp_arr_bbox)) if temp_arr_bbox_bools[i]]
            depth = [depth[i] for i in range(len(depth)) if temp_arr_bbox_bools[i]]
            return ids,  depth, temp_arr_bbox,temp_value_list,bbox

    def stop(self):
        if self.running:
            self.running = False
