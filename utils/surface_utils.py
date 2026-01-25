import datetime
import os
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import QLabel, QTableWidgetItem, QAbstractItemView, QHeaderView, QSizePolicy
from rtkb_sdk.RtNet import RtNet

from utils.log import LOG
from utils.display import DisplayFrame
from utils.surface_style import DarkSurfaceStyle
from utils.qthread_face_utils import FaceDetectQthread
from utils.qthread_temp_utils import TemperatureCalibration
from utils.qthread_rppg_utils import RPPGThread
# from utils.fun_utils import StreamCoordinatorDetectFace, StreamProcessor

from utils.camera_stream import RgbStream,TempStream
def mean_list(data: list):
    return sum(data) / (len(data) + 1e-8)


class SurfaceUtils(DarkSurfaceStyle):
    def __init__(self,
                 qmain_window,
                 config_data: dict | None = None,
                 logger_write = None
                 ):
        """
        可视化UI的Frame
        :param qmain_window: 主窗口
        :param config_data

        """
        super().__init__()
        self.max_history_rows = 20
        self.start_times = time.time()
        self.black_temp = None

        # 然后直接设置主窗口的样式
        self.setupUi(qmain_window)

        self.logger = LOG(self.textBrowser,logger_write)
        self.logger.log_info_enhanced('参数如下:')
        self.standard_info = config_data['BLACK']
        # 展示框
        IP = config_data['CAM_INFO']['cam_ip']
        self.mRtNet1 = RtNet()
        self.mRtNet2 = RtNet()

        iRet1 = self.mRtNet1.Init()
        iRet2 = self.mRtNet2.Init()
        # 当前代码为 模拟相机 后期修改为 直接 获取流
        self.rgb_stream = RgbStream(self.mRtNet1,IP.encode('utf-8'))
        self.temp_stream = TempStream(self.mRtNet2,IP.encode('utf-8'))

        # 数据可视化模块
        self.display = DisplayFrame(self, self.logger,config_data['BLACK'],config_data['DISPLAY_INFO'],config_data['EXPORT_INFO'])
        # 数据检测模块
        self.face_tracker = FaceDetectQthread(logger=self.logger, track_face=config_data['FACE'],deep_face=config_data['DEEP_FACE'])
        # 温度检测模块
        self.face_temp = TemperatureCalibration(logger=self.logger,config_data=config_data)
        # rppg模块
        self.rppgs = RPPGThread(logger=self.logger,config_data=config_data["RPPG"])

        self.display.signal.connect(self.add_warning_info)
        self.init_history_table()

    def init_history_table(self):
        """初始化历史记录表格，设置为只读"""

        # 设置表格为只读模式
        self.history_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 设置选择行为
        self.history_tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 整行选择
        self.history_tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)  # 单选

        # 设置列宽自适应
        header = self.history_tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)  # 所有列都拉伸填充

        # 设置表格属性
        self.history_tableWidget.setAlternatingRowColors(False)  # 交替行颜色
        self.history_tableWidget.setSortingEnabled(False)  # 禁用排序（如果需要可以开启）

    def apply_window_border(self, window):
        """直接设置主窗口的边框样式"""
        # 方法1: 通过样式表设置
        window.setStyleSheet(f"""
            QMainWindow {{
                border: 3px solid {self.bg_dark};
            }}
        """)

        # 方法2: 通过设置窗口背景
        palette = window.palette()
        palette.setColor(window.backgroundRole(), QColor(self.bg_dark))
        window.setPalette(palette)

        # 方法3: 设置窗口属性
        window.setAttribute(Qt.WA_TranslucentBackground, False)
    def add_warning_info(self,info):
        # {"time":time_str_time,"frame":rgb_frame,"temp":info['WARRING_MAX']
        # 将数据添加到history_tableWidget
        # 创建一个QTableWidgetItem

        """添加警告信息到表格，限制1000行，用户只读"""
        # {"time":time_str_time,"frame":rgb_frame,"temp":info['WARRING_MAX']
        time_data = str(info['time'])
        warning_temp = f"{info['temp']:.2f}"

        # 检查是否超过最大行数
        current_rows = self.history_tableWidget.rowCount()

        if current_rows >= self.max_history_rows:
            # 删除最旧的行（第一行）
            self.history_tableWidget.removeRow(0)
            # 更新当前行数
            current_rows = self.history_tableWidget.rowCount()

        # 在表格末尾添加新行
        row_position = current_rows
        self.history_tableWidget.insertRow(row_position)

        # 设置第一列数据
        item1 = QTableWidgetItem(time_data)
        item1.setTextAlignment(Qt.AlignCenter)
        item1.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # 只读，但可选择
        self.history_tableWidget.setItem(row_position, 0, item1)

        # 设置第二列数据
        item2 = QTableWidgetItem(warning_temp)
        item2.setTextAlignment(Qt.AlignCenter)
        item2.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # 只读，但可选择
        self.history_tableWidget.setItem(row_position, 1, item2)

        # 自动调整列宽
        # self.history_tableWidget.resizeColumnsToContents()

        # 自动滚动到最新行
        self.history_tableWidget.scrollToItem(item1, QAbstractItemView.PositionAtBottom)


        # 将图像v添加到显示模块face_display_label:QLable 并且贴合lable
        self._set_qlable_image(self.face_display_label,info['frame'])
        self.logger.log_info_enhanced(f"时间:{time_data} 出现 {warning_temp}°C")


    def star_carm(self):
        try:
            self.rgb_stream.start()
            self.temp_stream.start()
            time.sleep(0.5)
            self.face_tracker.start()
            time.sleep(0.5)

            self.face_temp.start()
            self.rppgs.start()
            self.logger.log_info_enhanced('视频流启动中...')
        except Exception as e:
            self.logger.log_info_enhanced(f'视频流启动失败 {e}')
            return


    def _set_qlable_image(self,qlable: QLabel, image: np.ndarray):
        """
        将numpy图像显示在QLabel上，并自适应QLabel大小
        """
        if image is None or image.size == 0:
            return False

        try:
            # 复制数据确保安全
            image = image.copy()
            height, width = image.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            # 保存QLabel的当前大小
            current_size = qlable.size()

            # 将图片缩放到QLabel的当前大小
            scaled_pixmap = pixmap.scaled(current_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 设置QLabel的尺寸策略为固定，防止它改变大小
            qlable.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            # 设置pixmap
            qlable.setPixmap(scaled_pixmap)

            # 设置QLabel的对齐方式，让图片居中显示
            qlable.setAlignment(Qt.AlignCenter)

            # 确保QLabel不会因为内容而改变大小
            qlable.setScaledContents(False)

            return True

        except Exception as e:
            self.logger.log_info_enhanced(f"设置图像失败: {e}","ERROR")
            return False

    def close_carm(self):
        try:
            # 视频流
            self.rgb_stream.stop()
            self.rgb_stream.wait()
            print("视频流停止")

            # 温度流
            self.temp_stream.stop()
            self.temp_stream.wait()
            print("温度流停止")
            # 关闭相机流
            self.mRtNet1.Exit()
            self.mRtNet2.Exit()
            print("相机流关闭")

            # 人脸检测 跟踪
            self.face_tracker.stop()
            self.face_tracker.wait()
            print('人脸检测 跟踪停止完成')
            # 人脸温度检测
            self.face_temp.stop()
            self.face_temp.wait()
            print("人脸温度检测停止")
            self.rppgs.stop()
            self.rppgs.wait()
            print("rppg 停止")
            self.logger.log_info_enhanced('视频流关闭中...')
        except Exception as e:
            try:
                self.mRtNet1.Exit()
                self.mRtNet2.Exit()
            except:
                self.logger.log_info_enhanced(f'关闭流失败: {e}')
            self.logger.log_info_enhanced(f'关闭流失败: {e}')
            return


