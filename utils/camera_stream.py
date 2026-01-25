import collections
import ctypes
import glob
import queue
import random
import threading
import time
from ctypes import *
import cv2
from PySide6.QtCore import QThread, Signal
from rtkb_sdk.RtNet import RtNet, TEMP_CALLBACK, JPEG_CALLBACK
from rtkb_sdk.Structures import *
import numpy as np



# 创建数据流 用于检测
rgb2face_queue = queue.Queue(maxsize=1)
gray2temp_queue = queue.Queue(maxsize=1)
# 创建 可视化数据流 用于可视化
rgb2display_queue = queue.Queue(maxsize=2)
gray2display_queue = queue.Queue(maxsize=2)

# -------------------------------------------------
# 创建信息传输队列
# 人脸box ids depth --> 温度流  [  "ids":list,"box":list,"depth":list  ]
face2temp_queue = queue.Queue(maxsize=1)
# 人脸box ids depth --> rppg流 [  "ids":list,"box":list,"depth":list "face":[arr1,arr2,],"miss":[int] ] # 其中miss表示需要删除的人脸信息
face2rppg_queue = queue.Queue(maxsize=1)
fps_2rppg_queue = queue.Queue(maxsize=1)
# 绘制显示
temp2display_queue = queue.Queue(maxsize=1) # "ids":[int,int,..] "temp":[list,list] "temp_smooth":[float,float,...]
rppg2display_queue = queue.Queue(maxsize=1) # "ids":[int,int,..] "bvp":[list]
face2display_queue = queue.Queue(maxsize=1) # "ids":[int,int,..] "bbox":[list]  "delete_ids":[int int]




info_dict = {}

"""
"ids":{
    "bbox":[x1,y1,x2,y2],
    "face":[arr1,arr2,arr3,],
    "miss":int,
    "depth":float,
    "temp_smooth":float,
    "temp":[float,float,...]
    }



"""

# 在全局区域定义
fps_window = collections.deque(maxlen=100)  # 保存最近30帧的时间戳
last_print_time = time.perf_counter()
print_interval = 1.0  # 每秒打印一次


def save_(current_time,frame):
    cv2.imwrite(f"./test_images/{current_time}.jpg", frame)

def queue_push(_render_rgb_queue, data):
    if _render_rgb_queue.full():
        try:
            _render_rgb_queue.get_nowait()  # 丢弃最旧帧
        except queue.Empty:
            pass
    _render_rgb_queue.put_nowait(data)

@JPEG_CALLBACK
def RgbJpegStreamCallback(paru8Data: POINTER(c_ubyte), u32DataLen: c_uint,
                          u32Width: c_uint, u32Height: c_uint,
                          u64Pts: c_ulonglong, pArg: c_void_p):
    """回调接收 RGB JPEG 数据"""
    global last_print_time

    numpy_array = np.ctypeslib.as_array(paru8Data, shape=(1, u32DataLen))
    img = cv2.imdecode(numpy_array, flags=cv2.IMREAD_COLOR)
    # 裁剪 ROI
    img = img[64:64 + 902, 402:402 + 1177]
    # img = img[64:64 + 902, 452:452 + 1177]
    current_time = time.perf_counter()  # 使用更高精度的计时器
    # t = threading.Thread(target=save_, args=(current_time,img))
    # t.start()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    queue_push(rgb2face_queue, {"image":img,"timestamp":time.perf_counter()})
    queue_push(rgb2display_queue, img)
    # 每秒打印一次FPS
    fps_window.append(current_time)
    # if len(fps_window) > 1:
    queue_push(fps_2rppg_queue,(len(fps_window) - 1) / (fps_window[-1] - fps_window[0]+1e-8))


@TEMP_CALLBACK
def TemperatureDataCallback(paru16Data: POINTER(c_ushort), u32Width: c_uint,
                            u32Height: c_uint, u64Pts: c_ulonglong, pArg: c_void_p):
    """回调接收温度数据"""
    s_array = np.ctypeslib.as_array(paru16Data, shape=(u32Height, u32Width))
    numpy_array = s_array * 0.1
    # if gray2temp_queue.full():
    #     try:
    #         gray2temp_queue.get_nowait()
    #     except queue.Empty:
    #         pass
    # gray2temp_queue.put_nowait(numpy_array)
    numpy_array = cv2.resize(numpy_array,(1177, 902))
    queue_push(gray2temp_queue, numpy_array)
    queue_push(gray2display_queue, numpy_array)


image_index = 0
img_list = glob.glob(r'/mnt/d/Projects/0_un_finish/rPPG-Toolbox/dataset_rppg/PURE/01-01/01-01/*.png')
st = time.time()

class RgbStream(QThread):
    signal = Signal()  # 轻量信号（不直接传 np.ndarray）
    time_signal = Signal(float)
    log_signal = Signal(str)

    def __init__(self, mRtNet: RtNet, pszServerIP):
        super(RgbStream, self).__init__()
        self.running = True
        self.RGB_STREAMER = None
        self.pszServerIP = pszServerIP
        self.mRtNet = mRtNet
        self.latest_rgb = None  # 缓存最新 RGB 帧
        self.star_time = None

    def run(self):
        self.RGB_STREAMER = self.mRtNet.StartRgbJpegStream(
            self.pszServerIP, RgbJpegStreamCallback, None
        )
        self.log_signal.emit('视频流启动完成')
        # global image_index, st
        # while self.running:
        #     try:
        #
        #         image_index += 1
        #         # if self.star_time is None:
        #         self.star_time = time.time()
        #         rgb_data = rgb2face_queue.get(timeout=1.0)
        #         st = time.time()
        #         print("rgb_data:", rgb_data["image"].shape)
        #         # st += 1./30.
        #         # rgb_data = cv2.imread(img_list[image_index%len(img_list)])
        #         # rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        #         # rgb_data = cv2.resize(rgb_data, (1920,1080))
        #         # queue_push(rgb2face_queue, {"image":rgb_data,"timestamp":st})
        #         queue_push(rgb2face_queue, rgb_data)
        #         queue_push(rgb2display_queue, rgb_data)
        #         time.sleep(0.03) # 对应 30 FPS
        #         self.time_signal.emit(time.time() - self.star_time)
        #         # print(f"FPS{1 / time.time() - self.star_time:.2f}")
        #     except queue.Empty:
        #         continue

        self.log_signal.emit('视频流停止完成')

    def stop(self):
        if self.running:
            self.log_signal.emit('视频流开始关闭')
            self.running = False
            # self.wait()
            self.log_signal.emit("视频流关闭完成")


class TempStream(QThread):
    temp_signal = Signal()  # 轻量信号
    log_signal = Signal(str)
    signal = Signal()
    def __init__(self, mRtNet: RtNet, pszServerIP):
        super(TempStream, self).__init__()
        self.TEMP_STREAMER = None
        self.running = True
        self.pszServerIP = pszServerIP
        self.mRtNet = mRtNet
        self.latest_temp = None  # 缓存最新温度帧
        self.star_time = None

    def run(self):
        self.TEMP_STREAMER = self.mRtNet.StartTemperatureStream(
            self.pszServerIP, TemperatureDataCallback, None
        )
        self.log_signal.emit('温度流启动完成')
        # while self.running:
        #     try:
        #         if self.star_time is None:
        #             self.star_time = time.time()
        #         # tmp_data = cv2.imread(img_list[image_index%len(img_list)],cv2.IMREAD_GRAYSCALE)
        #         tmp_data = gray2temp_queue.get(timeout=1.0)
        #         tmp_data = cv2.resize(tmp_data,(1177,902))
        #
        #         print("temp:",tmp_data.shape)
        #         queue_push(gray2temp_queue, tmp_data)
        #         queue_push(gray2display_queue, tmp_data)
        #         time.sleep(0.1)
        #         # self.latest_temp = tmp_data.copy()
        #         # self.temp_signal.emit()
        #         # self.signal.emit()
        #     except queue.Empty:
        #         continue

    def stop(self):
        if self.running:
            self.log_signal.emit('视频流开始关闭')
            self.running = False
            self.log_signal.emit("视频流关闭完成")


if __name__ == '__main__':
    from rtkb_sdk import RtNet as PyRtNet

    mRtNet = PyRtNet.RtNet()
    iRet = mRtNet.Init()
    if iRet != 0:
        mRtNet.Exit()
    # rgb = RgbStream(mRtNet, b"192.168.1.100")
    # rgb.start()
    while True:
        mRtNet.StartTemperatureStream(
            b"192.168.1.100", TemperatureDataCallback, None
        )
        star_time = time.time()
        tmp_data = gray2temp_queue.get(timeout=1.0)
        latest_temp = tmp_data.copy()
        print(f"FPS{1 / time.time() - star_time:.2f}")

