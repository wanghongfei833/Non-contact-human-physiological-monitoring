from models.RPPG.rppg_utils import RPPG_ONNX
from utils.camera_stream import *
from utils.log import LOG
class RPPGThread(QThread):
    signal = Signal(dict)

    def __init__(self,logger:LOG, config_data):
        super().__init__()
        """
          MOEL_PATH: './weights/rppg_10x3x72x72.onnx' # RPPG模型路径
          TIME_WINDOW: 30  # N帧检测一次 
          HR_TIME_THRESHOLD: 250 # 250帧 计算心率
          MIN_HR: 40
          MAX_HT: 150
        """
        # model_path, input_size = 72, time_window = 30, hr_thread = 250, min_hr = 45, max_hr = 150
        config_data['LOGGER'] = logger
        self.logger = logger
        self.muti_face_rppg = RPPG_ONNX(**config_data)
        self.logger.log_info_enhanced("心率检测模型加载成功...")

        # 定义基础信息 存储bvp
        self.running = True
        # 定义暂停信号
        self.pause = False

    def run(self):
        while self.running:
            try:
                try:
                    face_detect_info = face2rppg_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    fps = fps_2rppg_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                # face_detect_info
                # "id": tracker_id,
                # "face": [rgb_frame[b[1]:b[3], b[0]:b[2]] for b in bbox],
                # "delete_ids": del_ids,
                # 'timestamp': timestamp
                result = self.muti_face_rppg.predict(face_detect_info,fps)
                try:
                    self._send2display(result,face_detect_info['delete_ids'])
                except Exception as e:
                    self.logger.log_info_enhanced(f"qthread_rppg_utils.py-->RPPGThread-->run is error: {e}",'ERROR' )

            except Exception as e:
                print("qthread_rppg_utils.py-->RPPGThread-->run is error: ", e)
    @staticmethod
    def _send2display(send_info,del_ids):
        # 发送出去
        if len(send_info) == 0:
            pass
        send_data = {"ids":[],"bvp":[],"hr":[],"del_ids":del_ids}
        for k, v in send_info.items():
            send_data["ids"].append(k)
            send_data["bvp"].append(v['bvp'])
            send_data["hr"].append(v['hr'])
        queue_push(rppg2display_queue, send_data)

    def stop(self):
        if self.running:
            self.running = False
