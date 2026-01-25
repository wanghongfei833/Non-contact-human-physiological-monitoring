import os.path
from collections import deque
from datetime import datetime
from typing import Any, TypedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainterPath, QPen, QColor, QBrush, QFont
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsRectItem, QGraphicsEllipseItem
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem

from utils.camera_stream import *
from utils.log import LOG
from utils.smooth_value import smooth_temp_list
from utils.surface_style import DarkSurfaceStyle


# #  (1080, 1920, 3) --> []
# # (192, 256) -->(902,1177)


class GraphicsImageViewer(QObject):
    """
    优化的图形视图管理器
    支持在QGraphicsView中绘制图像、检测框和BVP折线图
    """

    def __init__(self, graphics_view: QGraphicsView, parent=None):
        super().__init__(parent)
        self.graphics_view = graphics_view

        # 初始化场景
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # 创建图片项
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # 存储检测框项
        self.bbox_items: Dict[str, List[QGraphicsRectItem]] = {}
        self.text_items: Dict[str, List[QGraphicsTextItem]] = {}
        self.chart_items: Dict[str, List[QGraphicsPathItem]] = {}  # 存储折线图项
        self.chart_backgrounds: Dict[str, List[QGraphicsRectItem]] = {}  # 存储折线图背景

        # 设置视图属性
        self.graphics_view.setRenderHint(QPainter.Antialiasing, True)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)

        # 当前图像尺寸
        self.current_width = 0
        self.current_height = 0

        # 折线图配置
        self.chart_config = {
            'width': 100,  # 折线图宽度
            'height': 50,  # 折线图高度
            'margin': 10,  # 与检测框的间距
            'line_color': QColor(0, 255, 0),  # 绿色
            'line_width': 2,
            'bg_color': QColor(0, 0, 0, 180),  # 半透明黑色背景
            'text_color': QColor(255, 255, 255),  # 白色文字
            'point_color': QColor(255, 255, 0),  # 黄色点
            'grid_color': QColor(100, 100, 100, 100),  # 灰色网格
            'show_grid': True,
            'show_points': True,
            'smooth': True,  # 平滑曲线
        }

        # 定时器用于延迟清除
        self.clear_timer = QTimer()
        self.clear_timer.setSingleShot(True)
        self.clear_timer.timeout.connect(self._clear_old_bboxes)

    def set_image(self, image_array: np.ndarray, convert_bgr: bool = True) -> bool:
        """
        设置图像到视图
        参数:
            image_array: numpy数组图像
            convert_bgr: 是否从BGR转换到RGB
        返回:
            bool: 是否成功
        """
        if image_array is None or image_array.size == 0:
            return False

        try:
            # 保存原始图像
            self.original_image = image_array.copy()

            # 处理图像格式
            if len(image_array.shape) == 2:
                # 灰度图
                self.current_height, self.current_width = image_array.shape
                q_image = QImage(image_array.data,
                                 self.current_width,
                                 self.current_height,
                                 self.current_width,
                                 QImage.Format_Grayscale8)
            elif len(image_array.shape) == 3:
                # 彩色图
                self.current_height, self.current_width, channels = image_array.shape

                if convert_bgr and channels == 3:
                    # OpenCV BGR 转 RGB
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * self.current_width
                    q_image = QImage(image_rgb.data,
                                     self.current_width,
                                     self.current_height,
                                     bytes_per_line,
                                     QImage.Format_RGB888)
                elif channels == 3:
                    # 已经是RGB
                    bytes_per_line = 3 * self.current_width
                    q_image = QImage(image_array.data,
                                     self.current_width,
                                     self.current_height,
                                     bytes_per_line,
                                     QImage.Format_RGB888)
                elif channels == 4:
                    # RGBA
                    bytes_per_line = 4 * self.current_width
                    q_image = QImage(image_array.data,
                                     self.current_width,
                                     self.current_height,
                                     bytes_per_line,
                                     QImage.Format_RGBA8888)
                else:
                    return False
            else:
                return False

            # 检查对象是否还存在
            if not self.graphics_view or not self.graphics_view.isVisible():
                return False
            # 转换为QPixmap
            pixmap = QPixmap.fromImage(q_image)
            # 更新图片项
            self.pixmap_item.setPixmap(pixmap)

            # 更新场景矩形
            self.scene.setSceneRect(QRectF(pixmap.rect()))

            # 自动调整视图
            self.fit_view()

            return True

        except Exception as e:
            self.logger.log_info_enhanced(f"设置图像失败: {e}")
            return False

    def update_bbox_info(self, bbox_info: Dict[str, Dict]):
        """
        更新检测框信息，同时绘制BVP折线图
        """
        if not bbox_info:
            return

        try:
            # 清除旧的检测框、文本和折线图
            self.clear_bboxes()
            for bbox_id, info in bbox_info.items():
                bbox = info.get('bbox', [0, 0, 0, 0])
                bbox_text = info.get('info', '')
                bvp_data = info.get('bvp', [])  # 获取BVP数据

                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # 创建检测框项
                    rect_item = QGraphicsRectItem(x1, y1, bbox_width, bbox_height)

                    # 设置颜色
                    pen = QPen(QColor(0, 255, 0), 2)  # 绿色，2像素宽
                    rect_item.setPen(pen)

                    # 添加到场景
                    self.scene.addItem(rect_item)

                    # 保存到字典
                    if 'bbox' not in self.bbox_items:
                        self.bbox_items['bbox'] = []
                    self.bbox_items['bbox'].append(rect_item)

                    # 如果有文本，创建文本项 - 使用配置的字体大小
                    if bbox_text:
                        text_item = QGraphicsTextItem(bbox_text)
                        text_item.setPos(x1, y1 - 20)  # 在框上方显示

                        # 设置文本样式 - 使用配置的字体大小
                        bbox_font = QFont(self.chart_config['font_family'], self.chart_config['bbox_font_size'])
                        text_item.setFont(bbox_font)
                        text_item.setDefaultTextColor(Qt.white)

                        self.scene.addItem(text_item)

                        if 'text' not in self.bbox_items:
                            self.bbox_items['text'] = []
                        self.bbox_items['text'].append(text_item)

                    # 如果有BVP数据，绘制折线图
                    if bvp_data and len(bvp_data) > 1:
                        self._draw_bvp_chart(bbox_id, bvp_data, (x1, y1, x2, y2))

        except Exception as e:
            self.logger.log_info_enhanced(f"更新检测框失败: {e}")

    def _draw_bvp_chart(self, bbox_id: str, bvp_data: List[float], bbox: Tuple[int, int, int, int]):
        """
        在检测框旁绘制BVP折线图
        参数:
            bbox_id: 目标ID
            bvp_data: BVP数据列表
            bbox: 检测框坐标 (x1, y1, x2, y2)
        """
        try:
            x1, y1, x2, y2 = bbox
            chart_width = self.chart_config['width']
            chart_height = self.chart_config['height']
            margin = self.chart_config['margin']

            # 计算折线图位置（优先放在右侧，如果空间不够则放在左侧）
            chart_x, chart_y = self._calculate_chart_position(x1, y1, x2, y2, chart_width, chart_height, margin)

            # 绘制折线图背景
            background = QGraphicsRectItem(chart_x, chart_y, chart_width, chart_height)
            background.setBrush(QBrush(self.chart_config['bg_color']))
            background.setPen(Qt.NoPen)
            self.scene.addItem(background)

            # 保存背景
            if 'chart_bg' not in self.chart_backgrounds:
                self.chart_backgrounds['chart_bg'] = []
            self.chart_backgrounds['chart_bg'].append(background)

            # 绘制边框
            border = QGraphicsRectItem(chart_x, chart_y, chart_width, chart_height)
            border.setPen(QPen(self.chart_config['line_color'], 1))
            self.scene.addItem(border)

            if 'chart_border' not in self.chart_items:
                self.chart_items['chart_border'] = []
            self.chart_items['chart_border'].append(border)

            # 计算数据点
            points = self._calculate_chart_points(bvp_data, chart_x, chart_y, chart_width, chart_height)

            if len(points) < 2:
                return

            # 创建折线路径
            path = QPainterPath()
            if self.chart_config['smooth'] and len(points) >= 3:
                # 平滑曲线
                self._create_smooth_path(path, points)
            else:
                # 直线连接
                path.moveTo(points[0])
                for i in range(1, len(points)):
                    path.lineTo(points[i])

            # 创建折线图项
            chart_item = QGraphicsPathItem(path)
            pen = QPen(self.chart_config['line_color'], self.chart_config['line_width'])
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            chart_item.setPen(pen)
            self.scene.addItem(chart_item)

            # 保存折线图
            if 'chart_line' not in self.chart_items:
                self.chart_items['chart_line'] = []
            self.chart_items['chart_line'].append(chart_item)

            # 绘制数据点
            if self.chart_config['show_points']:
                self._draw_chart_points(points)

                # 添加标题
                # 添加标题 - 使用配置的字体大小
                title = f"ID: {bbox_id}"
                title_item = QGraphicsTextItem(title)
                title_font = QFont(self.chart_config['font_family'], self.chart_config['title_font_size'])
                title_item.setFont(title_font)
                title_item.setDefaultTextColor(self.chart_config['text_color'])
                title_item.setPos(chart_x + 5, chart_y + 5)
                self.scene.addItem(title_item)

                if 'chart_text' not in self.chart_items:
                    self.chart_items['chart_text'] = []
                self.chart_items['chart_text'].append(title_item)

                # 如果数据足够，显示当前值 - 使用配置的字体大小
                if bvp_data:
                    current_value = bvp_data[-1]
                    value_text = f"{current_value:.3f}"
                    value_item = QGraphicsTextItem(value_text)
                    value_font = QFont(self.chart_config['font_family'], self.chart_config['value_font_size'])
                    value_font.setBold(True)
                    value_item.setFont(value_font)
                    value_item.setDefaultTextColor(self.chart_config['line_color'])

                    # 将值显示在折线图右上角
                    value_rect = value_item.boundingRect()
                    value_item.setPos(chart_x + chart_width - value_rect.width() - 5, chart_y + 5)
                    self.scene.addItem(value_item)

                    if 'chart_text' not in self.chart_items:
                        self.chart_items['chart_text'] = []
                    self.chart_items['chart_text'].append(value_item)

        except Exception as e:
            self.logger.log_info_enhanced(f"绘制BVP折线图失败: {e}")

    def _calculate_chart_position(self, x1: int, y1: int, x2: int, y2: int,
                                  chart_width: int, chart_height: int, margin: int) -> Tuple[int, int]:
        """
        计算折线图位置
        返回: (x, y) 坐标
        """
        # 优先放在检测框右侧
        chart_x = x2 + margin
        chart_y = y1

        # 检查是否超出图像边界
        if chart_x + chart_width > self.current_width:
            # 放在左侧
            chart_x = x1 - chart_width - margin
            if chart_x < 0:
                chart_x = x1
                chart_y = y2 + margin

                # 如果放在下方也超出边界，放在上方
                if chart_y + chart_height > self.current_height:
                    chart_y = y1 - chart_height - margin
                    if chart_y < 0:
                        chart_y = 0

        # 确保不超出边界
        chart_x = max(0, min(chart_x, self.current_width - chart_width))
        chart_y = max(0, min(chart_y, self.current_height - chart_height))

        return int(chart_x), int(chart_y)

    def _calculate_chart_points(self, data: List[float],
                                chart_x: float, chart_y: float,
                                chart_width: float, chart_height: float) -> List[QPointF]:
        """
        计算数据点在折线图中的坐标
        """
        if len(data) < 2:
            return []

        # 计算数据范围
        valid_data = [x for x in data if x is not None]
        if not valid_data:
            return []

        data_min = min(valid_data)
        data_max = max(valid_data)
        data_range = data_max - data_min

        if data_range == 0:
            data_range = 1  # 避免除零

        points = []
        x_step = chart_width / (len(data) - 1) if len(data) > 1 else 0

        for i, value in enumerate(data):
            if value is None:
                continue

            # X坐标
            x = chart_x + i * x_step

            # Y坐标（反转，因为Qt坐标原点在左上角）
            normalized = (value - data_min) / data_range
            y = chart_y + chart_height - (normalized * chart_height)

            points.append(QPointF(x, y))

        return points

    @staticmethod
    def _create_smooth_path(path: QPainterPath, points: List[QPointF]):
        """
        创建平滑的贝塞尔曲线路径
        """
        if len(points) < 2:
            return

        # 移动到第一个点
        path.moveTo(points[0])

        if len(points) == 2:
            # 只有两个点，直接连接
            path.lineTo(points[1])
        else:
            # 使用贝塞尔曲线平滑连接
            for i in range(1, len(points) - 1):
                # 控制点为前后点的中点
                control1 = QPointF(
                    (points[i - 1].x() + points[i].x()) / 2,
                    (points[i - 1].y() + points[i].y()) / 2
                )
                control2 = QPointF(
                    (points[i].x() + points[i + 1].x()) / 2,
                    (points[i].y() + points[i + 1].y()) / 2
                )

                path.cubicTo(control1, control2, points[i])

            # 最后一个线段
            path.lineTo(points[-1])

    def _draw_chart_points(self, points: List[QPointF]):
        """
        绘制数据点
        """
        for point in points:
            point_item = QGraphicsEllipseItem(
                point.x() - 2, point.y() - 2, 4, 4
            )
            point_item.setBrush(QBrush(self.chart_config['point_color']))
            point_item.setPen(Qt.NoPen)
            self.scene.addItem(point_item)

            if 'chart_points' not in self.chart_items:
                self.chart_items['chart_points'] = []
            self.chart_items['chart_points'].append(point_item)

    def clear_bboxes(self):
        """清除所有检测框、文本和折线图"""
        # 清除检测框
        for item_type, items in self.bbox_items.items():
            for item in items:
                if item.scene():
                    self.scene.removeItem(item)
        self.bbox_items.clear()

        # 清除折线图
        for item_type, items in self.chart_items.items():
            if isinstance(items, list):
                for item in items:
                    if item and item.scene():
                        self.scene.removeItem(item)
        self.chart_items.clear()

        # 清除折线图背景
        for item_type, items in self.chart_backgrounds.items():
            for item in items:
                if item.scene():
                    self.scene.removeItem(item)
        self.chart_backgrounds.clear()

    def _clear_old_bboxes(self):
        """清除旧的检测框（延迟执行）"""
        self.clear_bboxes()

    def clear(self):
        """清空图像"""
        self.pixmap_item.setPixmap(QPixmap())
        self.clear_bboxes()
        self.scene.setSceneRect(QRectF())
        self.current_width = 0
        self.current_height = 0

    def fit_view(self):
        """调整视图以适应图像"""
        if not self.pixmap_item.pixmap().isNull():
            self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def enable_zoom_drag(self, enable: bool = True):
        """启用/禁用缩放和拖动"""
        if enable:
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.graphics_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        else:
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)

    def get_current_image(self) -> Optional[np.ndarray]:
        """获取当前显示的图像"""
        return getattr(self, 'original_image', None)

    def set_chart_config(self, config: Dict):
        """设置折线图配置"""
        self.chart_config.update(config)


class DisplayInfo(TypedDict):
    depth: int
    bbox: List[int]
    temp: deque[float]
    bvp: list[float]
    hr: float
    temp_smooth: float
    WARRING: int
    WARRING_MAX: float
    loss_count: int


def expand_bbox(bbox, shape):
    """
    扩展bbox，但避免超过shape边界
    参数:
        bbox: [x1, y1, x2, y2] 格式的边界框
        shape: (height, width) 或 (height, width, channels) 格式的图像形状
    返回:
        扩展后的bbox [x1, y1, x2, y2]
    """
    h, w = shape[0], shape[1]  # 获取图像高度和宽度

    # 解析原始bbox
    x1, y1, x2, y2 = map(int, bbox)

    # 计算bbox的宽度和高度
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # 计算扩展后的尺寸（扩展为原来的1.2倍）
    expand_ratio = 1.2
    new_bbox_w = int(bbox_w * expand_ratio)
    new_bbox_h = int(bbox_h * expand_ratio)

    # 计算中心点
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # 计算扩展后的坐标
    half_w = new_bbox_w // 2
    half_h = new_bbox_h // 2

    new_x1 = center_x - half_w
    new_y1 = center_y - half_h
    new_x2 = center_x + half_w
    new_y2 = center_y + half_h

    # 确保边界不超出图像范围
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(w, new_x2)
    new_y2 = min(h, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]


class DisplayFrame(QObject):
    """
    优化的显示管理器
    使用QGraphicsView进行可视化，支持异步更新
    """
    # 定义信号
    update_completed = Signal()  # 更新完成信号
    signal = Signal(dict)

    def __init__(self,
                 qmain_window: 'DarkSurfaceStyle',
                 logger: 'LOG',
                 shared_standard: dict,
                 display_info: dict,
                 export_info: dict,
                 style=None
                 ):
        super().__init__()
        self.export_info = export_info
        self.display_info = display_info
        self.shared_standard = shared_standard
        self.logger = logger
        self.mw = qmain_window
        # 初始化图形视图管理器
        for k, v in shared_standard.items():
            self.logger.log_info_enhanced(f"标准黑体信息: {k}: {v}")
        self.rgb_viewer = GraphicsImageViewer(self.mw.graphicsViewImage)
        self.temp_viewer = GraphicsImageViewer(self.mw.graphicsViewGrays)
        # 自定义折线图样式
        self._set_canvs_config(style)
        # 共享内存缓冲区
        self.shared_detections: Dict[int, DisplayInfo] = {}

        # 锁
        self.data_lock = threading.RLock()

        # 更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._process_updates)
        self.update_timer.start(1 / self.display_info['HZ'])  # ~60Hz

    # ==================== 类数据更新函数 ====================
    def _set_canvs_config(self, style=None):
        if style is None:
            style = {
                'width': 150,  # 折线图宽度
                'height': 60,  # 折线图高度
                'margin': 10,  # 与检测框的间距
                'line_color': QColor(0, 255, 0),  # 绿色
                'line_width': 2,
                'bg_color': QColor(0, 0, 0, 180),  # 半透明黑色背景
                'text_color': QColor(0, 0, 255),  # 白色文字
                'point_color': QColor(255, 255, 0),  # 黄色点
                'grid_color': QColor(100, 100, 100, 100),  # 灰色网格
                'show_grid': True,
                'show_points': False,
                'smooth': True,  # 平滑曲线
                # 新增字体大小配置
                'title_font_size': 20,  # 标题字体大小
                'value_font_size': 20,  # 数值字体大小
                'bbox_font_size': 30,  # 检测框文本字体大小
                'font_family': 'Arial',  # 字体家族
            }
        self.rgb_viewer.set_chart_config(style)
        self.temp_viewer.set_chart_config(style)

    @staticmethod
    def _get_rgb_frame():
        # 获取RGB最新帧
        rgb_frame = None
        while True:
            try:
                rgb_frame = rgb2display_queue.get_nowait()
            except queue.Empty:
                break
        return rgb_frame

    @staticmethod
    def _get_temp_frame():
        temp_frame = None
        while True:
            try:
                temp_frame = gray2display_queue.get_nowait()
            except queue.Empty:
                break
        return temp_frame

    def _get_rppg_info(self):
        rppg_info = None
        while True:
            try:
                rppg_info = rppg2display_queue.get_nowait()

            except queue.Empty:
                break
        if rppg_info is not None:
            # ["ids"].append(k)["bvp"].append(v['bvp'])["hr"].append(v['hr'])
            # {"ids":ids,"temp":temp_value_list,"bbox":bbox,"depth":depth}
            for i, b, h in zip(rppg_info['ids'], rppg_info['bvp'], rppg_info['hr']):
                if i in self.shared_detections:
                    self.shared_detections[i]['bvp'] = b
                    self.shared_detections[i]['hr'] = h
                else:  # 没在 则是表明人脸已经消失
                    continue

    def _get_temp_info(self):
        temp_info = None
        while True:
            try:
                temp_info = temp2display_queue.get_nowait()
            except queue.Empty:
                break
        if temp_info is not None:
            # {"ids":ids,"temp":temp_value_list,"bbox":bbox,"depth":depth}
            for i, t, b, d in zip(temp_info['ids'], temp_info['temp'], temp_info['bbox'], temp_info['depth']):
                if i in self.shared_detections:
                    self.shared_detections[i]['temp'].append(t)
                    smooth_temp = smooth_temp_list(self.shared_detections[i]['temp'],
                                                   self.display_info['MIN_TEMP'],
                                                   self.display_info['MAX_TEMP'])
                    self.shared_detections[i]['temp_smooth'] = smooth_temp
                    #  =====================报警判断 ========================
                    if smooth_temp > self.export_info['MAX_TEMP_THRESHOLD']:
                        self.shared_detections[i]['WARRING'] += 1
                        self.shared_detections[i]['WARRING_MAX'] = max(self.shared_detections[i]['WARRING_MAX'], smooth_temp)
                    self.shared_detections[i]['bbox'] = b
                    self.shared_detections[i]['depth'] = d

    def get_face_detect_info(self):
        face_detect_info = None
        while True:
            try:
                face_detect_info = face2display_queue.get_nowait()
                # {'ids':tracker_id, 'depth_value':depth_value, 'delete_ids':del_ids,"bbox":bbox}
            except queue.Empty:
                break
        if face_detect_info is not None:
            # 首先使用    delete_ids 删除消失的人脸信息
            for ids in face_detect_info['delete_ids']:
                print(ids,"remove ...")
                del self.shared_detections[ids]
            for ids, box, depth in zip(face_detect_info['ids'], face_detect_info['bbox'], face_detect_info['depth_value']):
                if ids not in self.shared_detections:
                    self.shared_detections[ids] = {}
                    # 第一次出现的人脸 初始化温度窗口信息,因为 temp 为 append形式，其他数据都是直接
                    self.shared_detections[ids]['temp'] = deque(maxlen=self.display_info['TQL'])  # 初始化一下信息
                    self.shared_detections[ids]['WARRING'] = 0
                    self.shared_detections[ids]['WARRING_MAX'] = 0
                    self.shared_detections[ids]['loss_count'] = 0

                # 更新可视化类的 单数据 (int float )
                self.shared_detections[ids]['depth'] = depth
                self.shared_detections[ids]['bbox'] = box
    # ==================== 核心显示逻辑 ====================

    def _process_updates(self):
        """处理更新（在定时器中调用）"""
        try:
            # 获取当前数据
            rgb_frame = self._get_rgb_frame()
            temp_frame = self._get_temp_frame()
            self.get_face_detect_info()  # 最优先的一步 直接拿到对应的 人脸框和 深度信息。
            # 更新 温度数据
            try:
                self._get_temp_info()
            except Exception as e:
                self.logger.log_info_enhanced(f"温度数据更新失败:{e}","ERROR")
            try:
                self._get_rppg_info()
            except Exception as e:
                self.logger.log_info_enhanced(f"rppg 更新失败::{e}","ERROR")

            try:
                if rgb_frame is not None:
                    self._save_crop_rgb(rgb_frame)
            except Exception as e:
                self.logger.log_info_enhanced(f"保存截图失败::{e}","ERROR")

            # 如果有数据，进行处理
            if rgb_frame is not None or temp_frame is not None:
                self._display_frames(rgb_frame, temp_frame, self.shared_detections, self.shared_standard)

        except Exception as e:
            self.logger.log_info_enhanced(f"处理更新失败: :{e}","ERROR")

    def _save_crop_rgb(self, rgb_frame):
        # =============== 保存截图 ==================
        for i, info in self.shared_detections.items():
            if info['WARRING'] > 0 and info['WARRING'] > self.export_info['WARRING_THRESHOLD'] and info['WARRING']!=-999:
                # 存图
                bbox = info['bbox']
                # 将bbox扩展一下 但是避免越过rgb_frame的边界
                bbox = expand_bbox(bbox, rgb_frame.shape)
                # 使用时间作为数据命名
                # 创建年月日文件夹
                save_root = os.path.join(self.export_info['SAVE_PATH'], time.strftime("%Y-%m-%d"))
                os.makedirs(save_root, exist_ok=True)
                # 获取当前时间戳格式为 年月日+时分秒
                now = datetime.now()
                formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

                save_name_time = os.path.join(save_root, time.strftime("%H-%M-%S") + ".jpg")
                # 利用 box的xyxy将frame切割然后存储
                rgb_frame = rgb_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                bgr_frame = cv2.resize(bgr_frame, (256, 256))
                cv2.putText(bgr_frame, f"{info['WARRING_MAX']:.2f} C", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 2)
                cv2.imwrite(save_name_time, bgr_frame)
                info['WARRING'] = -999
                self.signal.emit({"time": formatted_time, "frame": rgb_frame, "temp": info['WARRING_MAX']})


    def _display_temp_(self, temp_frame, standard_info, detections):
        # 处理温度图像
        if temp_frame is not None:
            # 归一化温度数据
            temp_norm = self.norm_temp_data(temp_frame)
            if temp_norm is not None:
                # 准备绘制信息
                temp_bbox_info = {}
                # 添加标准黑体
                if standard_info:
                    # shared_standard {'standard_bbox': [938, 114, 150, 150], 'standard_temp': 35, 'standard_depth': 0.08}
                    bbox = standard_info.get('standard_bbox', [0, 0, 0, 0])
                    if len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        temp_bbox_info['standard'] = {
                            'bbox': [x, y, x + w, y + h],
                            'info': f"标准: {standard_info.get('standard_temp', 0.0):.1f}°C"
                        }

                # 添加检测框
                for det_id, det_info in detections.items():
                    temp_bbox = det_info.get('bbox')
                    if temp_bbox and len(temp_bbox) >= 4:
                        temp_bbox_info[str(det_id)] = {
                            'bbox': temp_bbox,
                            'info': self._format_detection_text(det_info, include_hr=False)
                        }

                # 更新温度视图
                self.temp_viewer.set_image(temp_norm, convert_bgr=False)
                if temp_bbox_info:
                    self.temp_viewer.update_bbox_info(temp_bbox_info)

    def _display_rgb_(self, rgb_frame, detections):
        # 处理RGB图像
        if rgb_frame is not None:
            # 准备绘制信息
            bbox_info = {}
            for det_id, det_info in detections.items():
                rgb_bbox = det_info.get('bbox')
                if rgb_bbox and len(rgb_bbox) >= 4:
                    bbox_info[str(det_id)] = {
                        'bbox': rgb_bbox,
                        'bvp': det_info['bvp'] if 'bvp' in det_info else [],
                        'info': self._format_detection_text(det_info, det_id=det_id)
                    }

            # 更新RGB视图
            self.rgb_viewer.set_image(rgb_frame, convert_bgr=False)
            if bbox_info:
                self.rgb_viewer.update_bbox_info(bbox_info)

    def _display_frames(self, rgb_frame: Optional[np.ndarray],
                        temp_frame: Optional[np.ndarray],
                        detections: Dict[int, DisplayInfo],
                        standard_info: Dict[str, Any]):
        """显示图像"""
        try:
            self._display_rgb_(rgb_frame, detections)
            self._display_temp_(temp_frame, standard_info, detections)

            # 发射更新完成信号
            self.update_completed.emit()

        except Exception as e:
            # self.logger.log_info_enhanced(f"显示图像失败: {e}",)
            self.logger.log_info_enhanced(f"显示图像失败: :{e}","ERROR")

    @staticmethod
    def _format_detection_text(det_info: Dict, include_hr: bool = True, det_id="") -> str:
        """格式化检测信息文本"""
        try:
            temp = det_info.get('temp_smooth', 0.0)
            if isinstance(temp, (list, tuple)):
                temp = temp[0] if temp else 0.0

            depth = det_info.get('depth', 0.0)
            if isinstance(depth, (list, tuple)):
                depth = depth[0] if depth else 0.0

            hr = det_info.get('hr', 0.0)

            # 构建文本
            parts = [
                f"ID: {det_id}",
                f"T:{temp:.1f}°C",
                f"D:{depth:.1f}m",
                # f"Δ:{(temp - 36.9):.3f}"
            ]

            if include_hr:
                parts.append(f"HR:{hr:.1f}")

            return "\n".join(parts)

        except Exception as e:
            return f"Error: {e}"

    # ==================== 工具方法 ====================

    @staticmethod
    def norm_temp_data(data: np.ndarray) -> Optional[np.ndarray]:
        """温度数据归一化"""
        if data is None or data.size == 0:
            return None

        try:
            # 归一化到0-255
            data_min = np.min(data)
            data_max = np.max(data)

            if data_max - data_min > 0:
                norm_data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                norm_data = np.zeros_like(data, dtype=np.uint8)

            # 应用伪彩色
            norm_data = cv2.applyColorMap(norm_data, cv2.COLORMAP_JET)

            # OpenCV是BGR，Qt需要RGB
            norm_data = cv2.cvtColor(norm_data, cv2.COLOR_BGR2RGB)

            return norm_data

        except Exception as e:
            print(f"温度数据归一化失败: {e}")
            return None

    # ==================== 公共接口 ====================

    def display(self, rgb_frame: Optional[np.ndarray] = None,
                temp_frame: Optional[np.ndarray] = None,
                draw_info: Optional[Dict[int, Dict]] = None,
                standard_info: Optional[Dict[str, Any]] = None):
        """
        显示图像（兼容旧接口）

        参数:
            rgb_frame: RGB图像
            temp_frame: 原始温度数据
            draw_info: 绘制信息
            standard_info: 标准黑体信息
        """
        with self.data_lock:
            if rgb_frame is not None:
                self.shared_rgb = rgb_frame.copy()
            if temp_frame is not None:
                self.shared_temp = temp_frame.copy()
            if draw_info is not None:
                self.shared_detections = draw_info.copy()
            if standard_info is not None:
                self.shared_standard = standard_info.copy()

    def clear_displays(self):
        """清空显示"""
        self.rgb_viewer.clear()
        self.temp_viewer.clear()
        with self.data_lock:
            # self.shared_rgb = None
            # self.shared_temp = None
            self.shared_detections.clear()
            self.shared_standard.clear()

    def fit_views(self):
        """调整视图适应图像"""
        self.rgb_viewer.fit_view()
        self.temp_viewer.fit_view()

    def enable_interaction(self, enable: bool = True):
        """启用/禁用用户交互（缩放/拖动）"""
        self.rgb_viewer.enable_zoom_drag(enable)
        self.temp_viewer.enable_zoom_drag(enable)

    def get_current_images(self) -> tuple:
        """获取当前显示的图像"""
        rgb = self.rgb_viewer.get_current_image()
        temp = self.temp_viewer.get_current_image()
        return rgb, temp
