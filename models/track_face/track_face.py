import onnxruntime as ort
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.base_utils import *
from models.track_face.process_utils import *

# 人脸跟踪
class Track_Face(object):
    def __init__(self, config_data_face):
        """
        初始化函数
        :param source_weights_path: 人脸识别模型
        :param confidence_threshold: 置信度阈值
        :param iou_threshold: iou阈值
        :return:
        """
        ### 人脸跟踪相关实例
        self.confidence_threshold = config_data_face['face_conf']
        self.iou_threshold = config_data_face['face_iou']
        # self.model = YOLO(config_data['face_weight'],task="detect")  # 加载人脸模型

        self.onnx_model = ort.InferenceSession(
            config_data_face['face_weight'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],

        )
        # input_shape = self.onnx_model.get_inputs()[0].shape   # 1x3x640x640
        self.input_name = self.onnx_model.get_inputs()[0].name
        self.output_name = self.onnx_model.get_outputs()[0].name
        input_data = np.random.random((1,3,640,640)).astype(np.float16)

        rst = self.onnx_model.run(None,{self.input_name:input_data})[0]     # 1x5x8400




        print(rst.shape)
        self.tracker = DeepSort(max_age=config_data_face['max_age'],
                                n_init=config_data_face['n_init'],
                                nn_budget=config_data_face['nn_budget'])

        self.annotated_label_frame = None
        self.annotated_label_frame_infrared = None
    def face_onnx_infer(self,image):
        h, w = image.shape[:2]
        # 预处理
        input_data, scale, left, top = letterbox_resize(image, (640, 640), color=(0, 0, 0), pad_rst=True)
        input_data = (input_data / 255.0).astype(np.float16).transpose((2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        results_rgb = self.onnx_model.run(None, {self.input_name: input_data})[0]

        # 4. 后处理：提取检测框和置信度
        xywh_boxes_640, scores = process_onnx_output_fast(
            results_rgb,
            self.confidence_threshold,
            self.iou_threshold
        )

        # 5. 将坐标映射回原始图像
        xywh_boxes_original = scale_coords_letterbox(
            xywh_boxes_640, (h, w), scale, left, top
        )
        return xywh_boxes_original,scores

    def deepsort_track(self, image: np.ndarray,xywh_boxes_original, scores):
        # 6. 准备DeepSort输入
        detections = [
            ([float(box[0]), float(box[1]), float(box[2]), float(box[3])], float(conf), 0)
            for box, conf in zip(xywh_boxes_original, scores)
        ]
        # 7. 更新跟踪器
        if len(detections):
            tracks = self.tracker.update_tracks(detections, frame=image)
            # 过滤已确认的track
            confirmed_tracks = [track for track in tracks if track.is_confirmed()]

            # 然后使用快速列表推导式
            track_id = [int(tid.track_id) for tid in confirmed_tracks]
            track_bbox = [list(map(int, box.to_ltrb())) for box in confirmed_tracks]

        else:
            # 没有检测到人脸
            track_id = []
            track_bbox = []

        return track_id, track_bbox

    def face_track(self, image: np.ndarray):
        """
        人脸跟踪方法
        :param image: RGB图像
        :return: [(人脸编号、xyxy格式坐标)]
        """
        # 将RGB图像和红外图像统一成相同格式
        # h, w = image.shape[:2]
        # # 预处理
        # input_data, scale, left, top = letterbox_resize(image, (640, 640),color=(0,0,0),pad_rst=True)
        # input_data = (input_data/255.0).astype(np.float16).transpose((2, 0, 1))
        # input_data = np.expand_dims(input_data, axis=0)
        # results_rgb = self.onnx_model.run(None,{self.input_name:input_data})[0]
        #
        # # 4. 后处理：提取检测框和置信度
        # xywh_boxes_640, scores = process_onnx_output_fast(
        #     results_rgb,
        #     self.confidence_threshold,
        #     self.iou_threshold
        # )
        #
        # # 5. 将坐标映射回原始图像
        # xywh_boxes_original = scale_coords_letterbox(
        #     xywh_boxes_640, (h, w), scale, left, top
        # )
        xywh_boxes_original, scores = self.face_onnx_infer(image)
        track_id, track_bbox = self.deepsort_track(image,xywh_boxes_original, scores)
        return track_id, track_bbox

