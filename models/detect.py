import time
import queue

import numpy as np
from PySide6.QtCore import QThread, Signal, QMutexLocker, QMutex, QWaitCondition

from models.segment_black.segment_black import SegmentBlack
from models.track_face.track_face import Track_Face
from utils.log import LOG


class BaseQthread(QThread):
    logger_signal = Signal(str)  # 文本信号

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def start(self, *args, **kwargs):
        with QMutexLocker(self.mutex):
            self.running = True
        super().start(*args, **kwargs)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
            self.condition.wakeAll()
        self.requestInterruption()
        self.wait()


class SegmentBlackQthread(BaseQthread):
    black_signal = Signal()  # 轻量信号，不再直接传 list

    def __init__(self, time_sleep=0.1):
        super().__init__()
        self.black_segment_md = SegmentBlack()
        self.time_sleep = time_sleep
        self.data_queue = queue.Queue(maxsize=3)
        self.latest_black_bbox = None  # 缓存最新结果

    def add_data(self, data):
        """丢弃最旧帧模式"""
        with QMutexLocker(self.mutex):
            if self.data_queue.full():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
            self.data_queue.put_nowait(data.copy())
            self.condition.wakeOne()

    def run(self) -> None:
        while self.running:
            try:
                data = self.data_queue.get(timeout=0.5)
                try:
                    self.process_data(data)
                except Exception as e:
                    self.logger_signal.emit(f"黑体分割出错: {e}")
            except queue.Empty:
                continue  # 空闲等待

    def process_data(self, data):
        black_id_detect, black_bbox_detect = self.black_segment_md.infer(data)
        self.latest_black_bbox = black_bbox_detect

        self.black_signal.emit()


class FaceDetectQthread(BaseQthread):
    face_signal = Signal()  # 轻量信号

    def __init__(self, logger: LOG, config_data: dict):
        super().__init__()
        self.face_track_model = Track_Face(config_data_face=config_data)
        self.logger = logger
        self.data_queue = queue.Queue(maxsize=5)
        self.latest_face_result = None  # 缓存最新结果

    def add_data(self, frame):
        with QMutexLocker(self.mutex):
            if self.data_queue.full():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
            self.data_queue.put_nowait(frame.copy())
            self.condition.wakeOne()

    def run(self) -> None:
        while self.running:
            try:
                data = self.data_queue.get(timeout=0.5)
                try:
                    self.process_data(data)
                except Exception as e:
                    self.logger_signal.emit(f"人脸检测出错: {e}")
            except queue.Empty:
                with QMutexLocker(self.mutex):
                    self.condition.wait(self.mutex, 100)

    def process_data(self, rgb_frame):
        tracker_id, bbox = self.face_track_model.face_track(rgb_frame)
        # 计算距离
        face_info = {
            k: {
                "bbox": v,
                "face": rgb_frame[v[1]:v[3], v[0]:v[2]],

            }
            for k, v in zip(tracker_id, bbox)
        }
        self.latest_face_result = {
            'rgb': rgb_frame,
            'face_info': face_info
        }
        self.face_signal.emit()
