from typing import Dict, List, TypedDict

import cv2
import numpy as np
import onnxruntime as ort
from scipy import fft
from scipy import signal
from scipy.fft import fftfreq
from utils.log import LOG


def bvp2hr(bvp_signal, fps, min_hr=40, max_hr=200, method='welch'):
    """
    从BVP信号计算心率
    
    参数:
    bvp_signal -- BVP信号数组
    fps -- 采样频率 (Hz)
    min_hr -- 最小有效心率 (bpm), 默认40
    max_hr -- 最大有效心率 (bpm), 默认200
    method -- 计算方法: 'welch'(默认), 'fft', 'autocorr'
    
    返回:
    hr -- 计算得到的心率值 (bpm)
    """

    # 输入验证
    if len(bvp_signal) < 2:
        raise ValueError("BVP信号长度太短")
    # print(len(bvp_signal),times,fps)
    if fps <= 0:
        raise ValueError("采样频率必须大于0")

    # 预处理信号
    # bvp_clean = preprocess_bvp(bvp_signal, fps)

    # 根据选择的方法计算心率
    if method == 'welch':
        hr = hr_from_welch(bvp_signal, fps, min_hr, max_hr)
    elif method == 'fft':
        hr = hr_from_fft(bvp_signal, fps, min_hr, max_hr)
    elif method == 'autocorr':
        hr = hr_from_autocorr(bvp_signal, fps, min_hr, max_hr)
    else:
        raise ValueError("不支持的method参数，请选择 'welch', 'fft' 或 'autocorr'")

    return hr


def preprocess_bvp(bvp_signal, fps):
    """信号预处理"""
    # 去趋势
    detrended = signal.detrend(bvp_signal)

    # 带通滤波 (0.7-4 Hz, 对应42-240 bpm)
    nyquist = fps / 2
    lowcut = 0.7 / nyquist  # 对应42 bpm
    highcut = 4.0 / nyquist  # 对应240 bpm

    # 使用巴特沃斯滤波器
    b, a = signal.butter(4, [lowcut, highcut], btype='band')
    filtered = signal.filtfilt(b, a, detrended)

    return filtered


def hr_from_welch(bvp_signal, fps, min_hr, max_hr):
    """使用Welch方法计算心率"""
    # 计算功率谱密度
    _freqs, psd = signal.welch(bvp_signal, fps, nperseg=min(len(bvp_signal), 256))

    # 转换为bpm
    _freqs_bpm = _freqs * 60

    # 限制在有效心率范围内
    mask = (_freqs_bpm >= min_hr) & (_freqs_bpm <= max_hr)
    valid_freqs = _freqs_bpm[mask]
    valid_psd = psd[mask]

    if len(valid_freqs) == 0:
        raise ValueError("在有效心率范围内未找到明显的峰值")

    # 找到最大功率对应的频率
    peak_idx = np.argmax(valid_psd)
    hr = valid_freqs[peak_idx]

    return hr


def hr_from_fft(bvp_signal, fps, min_hr, max_hr):
    """使用FFT方法计算心率"""
    n = len(bvp_signal)

    # 计算FFT
    fft_result = fft(bvp_signal)
    freqs = fftfreq(n, 1 / fps)

    # 取正频率部分
    positive_freqs = freqs[:n // 2]
    magnitude = np.abs(fft_result[:n // 2])

    # 转换为bpm
    freqs_bpm = positive_freqs * 60

    # 限制在有效心率范围内
    mask = (freqs_bpm >= min_hr) & (freqs_bpm <= max_hr)
    valid_freqs = freqs_bpm[mask]
    valid_magnitude = magnitude[mask]

    if len(valid_freqs) == 0:
        raise ValueError("在有效心率范围内未找到明显的峰值")

    # 找到最大幅度对应的频率
    peak_idx = np.argmax(valid_magnitude)
    hr = valid_freqs[peak_idx]

    return hr


def hr_from_autocorr(bvp_signal, fps, min_hr, max_hr):
    """使用自相关方法计算心率"""
    # 计算自相关函数
    autocorr = np.correlate(bvp_signal, bvp_signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # 取正延迟部分

    # 找到第一个峰值后的主要峰值（跳过零延迟的峰值）
    peaks, _ = signal.find_peaks(autocorr[10:])  # 跳过前10个点避免零延迟峰值
    peaks += 10  # 调整索引

    if len(peaks) == 0:
        return -1
        # raise ValueError("未找到明显的自相关峰值")
    # 找到最高峰值
    main_peak = peaks[np.argmax(autocorr[peaks])]

    # 计算心率
    period = main_peak / fps  # 周期（秒）
    hr = 60 / period  # 转换为bpm

    # 检查心率是否在有效范围内
    if hr < min_hr or hr > max_hr:
        # 如果不在范围内，尝试找下一个峰值
        if len(peaks) > 1:
            peaks = peaks[peaks != main_peak]  # 移除当前峰值
            main_peak = peaks[np.argmax(autocorr[peaks])]
            period = main_peak / fps
            hr = 60 / period
        else:
            return -1

    return hr


class PersonInfo(TypedDict):
    face: List[np.ndarray]
    bvp: List[float]
    hr: float
    # bbox:list[float]
    time_list: List[float]


class RPPG_ONNX(object):
    def __init__(self, **kw):
        self.logger: LOG = kw.get('LOGGER')
        self.model_path = kw.get('MODEL_PATH')
        self.hr_time_threshold = kw.get('HR_TIME_THRESHOLD')
        self.min_hr = kw.get('MIN_HR')
        self.max_hr = kw.get('MAX_HR')
        self.time_window = kw.get('TIME_WINDOW')
        self.input_size = kw.get('INPUT_SIZE')
        # 180 帧--> 25FPS--> 7.2 S
        self.onnx_model = ort.InferenceSession(
            self.model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.input_name = self.onnx_model.get_inputs()[0].name
        self.output_name = self.onnx_model.get_outputs()[0].name
        input_data = np.random.randn(self.time_window + 1, 3, *self.input_size).astype(np.float32)
        self.onnx_model.run(None, {self.input_name: input_data})
        self.face_info: Dict[int, PersonInfo] = {}  # 用来记录每个IDS 对应的人脸信息和对应的bvp信息 {"ids":face:[arr1,arr2,...],bvp:[float],hr:float}
        self.pad_arr = np.zeros((1, 3, *self.input_size))
        self.fps = 30

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    def process_image(self, input_image: np.ndarray):
        image = self.standardized_data(input_image)
        return np.transpose(image, (0, 3, 1, 2))

    def predict(self, face_info, fps) -> Dict[int, PersonInfo]:
        # 预处理数据
        # face_detect_info:
        #   "id": tracker_id,
        #   "face": [rgb_frame[b[1]:b[3], b[0]:b[2]] for b in bbox],
        # "delete_ids": del_ids

        self._update_face_info(face_info)
        for ids, values in self.face_info.items():
            if values['face'] is not None and len(values['face']) < self.time_window:
                continue
            try:
                # 模型推理
                face_values = values['face']
                # print("执行一次推理",ids)
                face_values = np.array(face_values)
                face_values = self.standardized_data(face_values)
                input_arr = np.concatenate([face_values, face_values[-1:]], axis=0, dtype=np.float32)
            except Exception as e:
                self.logger.log_info_enhanced(f"models/RPPG/rppg_utils/RPPG_ONNX/predict 预处理出现错误{e}", "ERROR")
                continue
            # input_arr = np.concatenate((face_values,self.pad_arr),axis=0,dtype=np.float32)
            # 均值方差处理
            try:
                # print('rppg')
                _output = self.onnx_model.run(None, {'input': input_arr})[0]
                # print("rppg ok")
            except Exception as e:
                self.logger.log_info_enhanced("models/RPPG/rppg_utils/RPPG_ONNX/ONNX_RUN:{e}", "ERROR")
                _output = np.zeros(len(input_arr) - 1)

            # print("ids:",ids,"推理完成一次..")
            try:
                self.face_info[ids]['bvp'].extend(_output.flatten().tolist())
                # if len(self.face_info[ids]['bvp']) >= self.hr_time_threshold:  # 执行计算心率
                hr = bvp2hr(values['bvp'], fps, self.min_hr, self.max_hr, "autocorr")
                if self.min_hr <hr< self.max_hr:
                    values['hr'] = hr
                else:
                    print("异常", len(values['bvp']),end="\t")
                    values['bvp'] = values['bvp'][:-1 * _output.shape[0]]   # 清除异常的bvp信息
                    print(len(values['bvp']))
                values['bvp'] = values['bvp'][-self.hr_time_threshold:]
                values['time_list'] = values['time_list'][-self.hr_time_threshold:]
                values['face'] = []  # 清空数据
            except Exception as e:
                self.logger.log_info_enhanced(f"models/RPPG/rppg_utils/RPPG_ONNX/bvp2hr 计算错误{e}", "ERROR")
        return self.face_info

    def _update_face_info(self, face_info):
        ids = face_info['ids']
        face = face_info['face']
        delete_ids = face_info['delete_ids']
        timestamp = face_info['timestamp']
        try:
            for di in delete_ids:
                if di in self.face_info:
                    del self.face_info[di]
        except Exception as e:
            print("rppg_utils.py RPPG_ONNX --> _update_face_info-->del ids 出错:", e)
        # 添加数据
        try:
            for i, f in zip(ids, face):
                # 临时存储
                f = self._process_face(f)
                if i in self.face_info:
                    self.face_info[i]['face'].append(f)
                    self.face_info[i]['time_list'].append(timestamp)
                else:
                    self.face_info[i] = PersonInfo(
                        face=[f],
                        bvp=[],
                        hr=-1,
                        time_list=[timestamp],

                    )
        except Exception as e:
            print("rppg_utils.py RPPG_ONNX --> _update_face_info 出错:", e)

    def _process_face(self, face) -> np.ndarray:
        # face = letterbox_resize(face, new_size=self.input_size)
        face = cv2.resize(face, self.input_size)  # 模型训练时使用的时resize
        face = (face / 255.0)
        face = np.transpose(face, (2, 0, 1)).astype(np.float64)
        return face
