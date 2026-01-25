from models.track_face.track_face import Track_Face
from utils.camera_stream import *
from utils.log import LOG



class FaceDetectQthread(QThread):
    face_signal = Signal(str)  # 轻量信号
    signal = Signal(dict)
    def __init__(self, logger: LOG, track_face,deep_face):
        super().__init__()
        # 解析 人脸检测信息

        self.rgb_stream = None
        self.ids = []
        self.miss = {}
        self.logger = logger
        self.face_track_model = Track_Face(track_face)
        self.logger.log_info_enhanced(f'人脸检测模型加载完成...')
        self.mdc = track_face['MDC']

        self.running = True
   
        # ----- 定义人脸距离测量参数---
        # 定义人脸的宽度
        self.KNOWN_FACE_WIDTH = deep_face['KNOWN_FACE_WIDTH']
        W0 = deep_face['W0']
        D0 = deep_face['D0']
        self.f = (W0 * D0) / self.KNOWN_FACE_WIDTH
        # ===============定义 暂停信号 ==============================
        # 暂停信号
        self.pause = False

    def run(self) -> None:
        while self.running:
            if self.pause:
                time.sleep(0.1)
                continue
            # 关键步骤计时
            try:
                # 使用带超时的get，可以检查停止事件
                rgb_queue = rgb2face_queue.get(timeout=0.1)
            except queue.Empty:
                # 队列为空，继续循环
                continue
            rgb_frame = rgb_queue['image']
            timestamp = rgb_queue['timestamp']
            if rgb_frame is not None:
                try:
                    # start = time.time()
                    tracker_id, bbox = self.face_track_model.face_track(rgb_frame)
                    depth_value = self._computer_depth_rgb(np.array(bbox))
                    # 更新人脸信息 返回miss信息
                    miss = list(set(self.ids)-set(tracker_id))
                    del_ids = []
                    if len(miss) > 0:
                        # 删除部分人脸信息
                        for i in miss:
                            if self.miss[i] >= self.mdc:    # 最大次数后 移除人脸
                                self.ids.remove(i)
                                del_ids.append(i)
                            else:
                                self.miss[i] += 1
                    uptdate = list(set(tracker_id)-set(self.ids))
                    if len(uptdate) > 0:
                        for i in uptdate:
                            self.ids.append(i)
                            self.miss[i] = 0
                    depth_value = depth_value.tolist()
                except Exception as e:
                    self.logger.log_info_enhanced(f'人脸检测错误：{e}',"ERROR")
                    continue
                try:
                    # self.logger.log_info_enhanced(f'脸部检测FPS：{ 1/ (use_time+1e-8):.1f}')
                    # --------------- 发送信息 ----------
                    # 发送给display.py--> get_face_detect_info函数
                    queue_push(face2display_queue, {'ids':tracker_id, 'depth_value':depth_value, 'delete_ids':del_ids,"bbox":bbox})
                except Exception as e:
                    self.logger.log_info_enhanced(f'face2display_queue 发送信息错误：{e}',"ERROR")
                    
                try:
                    # 发送给 qthread_rppg_utils.py --> RPPGThread-->run 函数
                    queue_push(face2rppg_queue,{
                        "ids": tracker_id,
                        "face":[rgb_frame[b[1]:b[3],b[0]:b[2]] for b in bbox],
                        "delete_ids":del_ids,
                        'timestamp':timestamp
                    })
                except Exception as e:
                    self.logger.log_info_enhanced(f'face2rppg_queue 推送信息错误：{e}',"ERROR")
                try:
                    # 发送给 qthread_temp_utils.py --> TemperatureCalibration-->run函数
                    queue_push(face2temp_queue,{"id":tracker_id,"bbox":bbox,"depth":depth_value,"delete_ids":del_ids})
                except Exception as e:
                    self.logger.log_info_enhanced(f'qthread_temp_utils发送信息错误：{e}',"ERROR")


    def _computer_depth_rgb(self, bbox: np.ndarray) -> np.ndarray :
        """
        bbox --> [x,y1,x2,y2]
        """
        if len(bbox) == 0:
            return np.array([])
        try:
            # 计算 高宽比
            w = bbox[:, 2] - bbox[:, 0]
            h = bbox[:, 3] - bbox[:, 1]
            scale = w / h
            depth_value = (self.KNOWN_FACE_WIDTH * self.f) / w / 100

            depth_value[scale < 0.55] *= 0.95
            depth_value[scale > 0.65] *= 1.05
            return depth_value
        except Exception as e:
            return  np.array([])

    def stop(self):
        if self.running:
            self.running = False