from UI.video_suface import Ui_MainWindow

"""
主要对设计的surface进行 风格的修改
会在surface_utils中进行组件的功能撰写
随后在main中嗲用surface_utils
"""


# class DarkSurfaceStyle(Ui_MainWindow):
#     def __init__(self):
#         super(DarkSurfaceStyle, self).__init__()


import time
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QColor, QPainter
from PySide6.QtWidgets import (QGraphicsDropShadowEffect,
                               QAbstractItemView, QTableWidgetItem, QMainWindow)


class DarkSurfaceStyle(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setup_style()

    def setupUi(self, MainWindow):
        """重写setupUi方法，先调用父类方法再应用样式"""
        super().setupUi(MainWindow)
        # 移除这一行，保留系统标题栏
        # MainWindow.setWindowFlags(Qt.FramelessWindowHint)

        # 应用主题
        self.apply_dark_theme()
        self.apply_modern_effects()
        self.setup_fonts()

    def setup_style(self):
        """初始化样式设置"""
        self.primary_color = "#2E8B57"  # 海绿色作为主色调
        self.secondary_color = "#1E90FF"  # 道奇蓝作为辅助色
        self.accent_color = "#FF6347"  # 番茄红作为强调色
        self.bg_dark = "#1E1E2E"  # 深色背景
        self.bg_card = "#2D2D44"  # 卡片背景
        self.text_primary = "#FFFFFF"  # 主要文字
        self.text_secondary = "#B0B0B0"  # 次要文字
        self.border_color = "#404040"  # 边框颜色

    def apply_dark_theme(self):
        main_window = self.centralwidget.parent()
        """应用深色主题样式表"""
        style_sheet = f"""
        /* 中央部件 */
        QWidget#centralwidget {{
            background: transparent;
        }}

        /* 主框架样式 */
        QFrame {{
            background-color: {self.bg_card};
            border: 1px solid {self.border_color};
            border-radius: 12px;
        }}

        /* 图像显示框架 */
        #frame_4 {{
            background-color: {self.bg_dark};
            border: 2px solid {self.border_color};
            border-radius: 8px;
        }}

        /* 图形视图样式 */
        QGraphicsView {{
            background-color: {self.bg_dark};
            border: 1px solid {self.border_color};
            border-radius: 6px;
            padding: 2px;
        }}

        QGraphicsView:hover {{
            border: 1px solid {self.primary_color};
        }}

        /* 按钮样式 */
        QPushButton {{
            background-color: {self.primary_color};
            color: {self.text_primary};
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: bold;
            min-height: 20px;
        }}

        QPushButton:hover {{
            background-color: #3DA56D;
        }}

        QPushButton:pressed {{
            background-color: #26734D;
        }}

        /* 特殊按钮样式 */
        #config_button {{
            background-color: {self.secondary_color};
        }}

        #config_button:hover {{
            background-color: #2A9FFF;
        }}

        #history_button {{
            background-color: #6A5ACD;
        }}

        #history_button:hover {{
            background-color: #7B6CE5;
        }}

        #log_button {{
            background-color: #FF6347;
        }}

        #log_button:hover {{
            background-color: #FF7A63;
        }}

        /* 文本浏览器样式 */
        QTextBrowser {{
        background-color: {self.bg_dark};
        border: 1px solid {self.border_color};
        border-radius: 8px;
        color: #F0F8FF;  /* 修改这里：改为浅灰色，确保在深色背景上清晰可见 */
        font-size: 13px;
        padding: 8px;
        selection-background-color: {self.primary_color};
    }}

        /* 标签样式 */
        QLabel {{
            color: {self.text_primary};
            background: transparent;
            border: none;
        }}

        /* 抓拍标签特殊样式 */
        #face_display_label {{
            background-color: {self.bg_dark};
            border: 2px solid {self.primary_color};
            border-radius: 10px;
            color: {self.primary_color};
            font-weight: bold;
        }}

        /* 表格样式 */
        QTableWidget {{
            background-color: {self.bg_dark};
            border: 1px solid {self.border_color};
            border-radius: 8px;
            gridline-color: {self.border_color};
            color: {self.text_primary};
            font-size: 13px;
            selection-background-color: {self.primary_color};
        }}

        QTableWidget::item {{
            padding: 8px;
            border-bottom: 1px solid {self.border_color};
        }}

        QTableWidget::item:selected {{
            background-color: {self.primary_color};
            color: white;
        }}

        QHeaderView::section {{
            background-color: {self.primary_color};
            color: white;
            padding: 10px;
            border: none;
            font-weight: bold;
        }}

        /* 菜单栏样式 */
        QMenuBar {{
            background-color: {self.bg_card};
            color: {self.text_primary};
            border-bottom: 1px solid {self.border_color};
            padding: 5px;
        }}

        QMenuBar::item {{
            background-color: transparent;
            padding: 8px 15px;
            border-radius: 4px;
        }}

        QMenuBar::item:selected {{
            background-color: {self.primary_color};
        }}

        /* 滚动条样式 */
        QScrollBar:vertical {{
            background-color: {self.bg_dark};
            width: 15px;
            margin: 0px;
            border-radius: 7px;
        }}

        QScrollBar::handle:vertical {{
            background-color: {self.primary_color};
            border-radius: 7px;
            min-height: 30px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: #3DA56D;
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        """

        # 应用样式表到主窗口
        if main_window and isinstance(main_window, QMainWindow):
            main_window.setStyleSheet(style_sheet)
        else:
            # 回退到只设置centralwidget
            self.centralwidget.setStyleSheet(style_sheet)

    def apply_modern_effects(self):
        """应用现代视觉效果如阴影、动画等"""
        # 为框架添加阴影效果
        shadow_effects = [
            (self.frame, 20, 10),
            (self.frame_4, 15, 8),
            (self.frame_3, 15, 8),
            (self.frame_2, 10, 5)
        ]

        for widget, blur_radius, offset in shadow_effects:
            if hasattr(self, widget.objectName().replace("frame_", "")):
                shadow = QGraphicsDropShadowEffect()
                shadow.setBlurRadius(blur_radius)
                shadow.setXOffset(0)
                shadow.setYOffset(offset)
                shadow.setColor(QColor(0, 0, 0, 100))
                widget.setGraphicsEffect(shadow)

        # 为按钮添加更强的阴影效果
        button_shadows = [
            (self.config_button, 15, 5),
            (self.history_button, 15, 5),
            (self.log_button, 15, 5)
        ]

        for button, blur_radius, offset in button_shadows:
            if hasattr(self, button.objectName().replace("_button", "")):
                shadow = QGraphicsDropShadowEffect()
                shadow.setBlurRadius(blur_radius)
                shadow.setXOffset(0)
                shadow.setYOffset(offset)
                shadow.setColor(QColor(0, 0, 0, 80))
                button.setGraphicsEffect(shadow)

        # 设置表格属性
        if hasattr(self, 'history_tableWidget'):
            self.history_tableWidget.setAlternatingRowColors(True)
            self.history_tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 修正这里
            self.history_tableWidget.horizontalHeader().setStretchLastSection(True)

        # 设置文本浏览器属性
        if hasattr(self, 'textBrowser'):
            self.textBrowser.setOpenExternalLinks(True)

        # 设置图形视图属性
        if hasattr(self, 'graphicsViewImage'):
            self.graphicsViewImage.setRenderHint(QPainter.Antialiasing)
        if hasattr(self, 'graphicsViewGrays'):
            self.graphicsViewGrays.setRenderHint(QPainter.Antialiasing)

    def setup_fonts(self):
        """设置现代化字体"""
        # 尝试加载系统字体，回退到默认字体
        font_families = ["Segoe UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"]

        # 设置整体字体
        app_font = QFont()
        for font_family in font_families:
            if font_family in QFontDatabase.families():
                app_font.setFamily(font_family)
                break
        app_font.setPointSize(10)
        self.centralwidget.setFont(app_font)

        # 设置标题字体
        title_font = QFont(app_font)
        title_font.setPointSize(16)
        title_font.setBold(True)

        # 设置按钮字体
        button_font = QFont(app_font)
        button_font.setPointSize(11)
        button_font.setBold(True)

        if hasattr(self, 'config_button'):
            self.config_button.setFont(button_font)
        if hasattr(self, 'history_button'):
            self.history_button.setFont(button_font)
        if hasattr(self, 'log_button'):
            self.log_button.setFont(button_font)

        # 设置表格字体
        table_font = QFont(app_font)
        table_font.setPointSize(10)
        if hasattr(self, 'history_tableWidget'):
            self.history_tableWidget.setFont(table_font)

        # 设置标签字体
        label_font = QFont(app_font)
        label_font.setPointSize(20)
        label_font.setBold(True)
        if hasattr(self, 'face_display_label'):
            self.face_display_label.setFont(label_font)

    def update_ui_colors(self, primary_color=None, bg_color=None):
        """动态更新UI颜色（可选功能）"""
        if primary_color:
            self.primary_color = primary_color
        if bg_color:
            self.bg_dark = bg_color

        # 重新应用样式
        self.apply_dark_theme()


class SurfaceUtils(DarkSurfaceStyle):
    def __init__(self, qmain_window, config_data: dict | None = None):
        # 正确调用父类初始化
        super().__init__()

        self.max_history_rows = 20
        self.start_time = time.time()
        self.black_temp = None

        # 确保在调用setupUi之前，DarkSurfaceStyle已经完成初始化
        self.setupUi(qmain_window)

        # 如果有配置数据，可以在这里进行额外的初始化
        if config_data:
            self.apply_config(config_data)

    def apply_config(self, config_data: dict):
        """应用配置数据"""
        # 这里可以根据配置数据调整UI
        if 'primary_color' in config_data:
            self.update_ui_colors(primary_color=config_data['primary_color'])

        if 'max_history_rows' in config_data:
            self.max_history_rows = config_data['max_history_rows']

    def add_history_record(self, time_str: str, temperature: str):
        """添加历史记录到表格"""
        if not hasattr(self, 'history_tableWidget'):
            return

        row_position = self.history_tableWidget.rowCount()
        self.history_tableWidget.insertRow(row_position)

        self.history_tableWidget.setItem(row_position, 0, QTableWidgetItem(time_str))
        self.history_tableWidget.setItem(row_position, 1, QTableWidgetItem(temperature))

        # 限制历史记录数量
        if self.history_tableWidget.rowCount() > self.max_history_rows:
            self.history_tableWidget.removeRow(0)

    def update_face_display(self, status: str, color: str = None):
        """更新面部显示标签"""
        if not hasattr(self, 'face_display_label'):
            return

        self.face_display_label.setText(status)
        if color:
            # 动态更新标签颜色
            style = f"""
            #face_display_label {{
                background-color: {self.bg_dark};
                border: 2px solid {color};
                border-radius: 10px;
                color: {color};
                font-weight: bold;
            }}
            """
            self.face_display_label.setStyleSheet(style)

    def append_log(self, message: str):
        """添加日志信息"""
        if not hasattr(self, 'textBrowser'):
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.textBrowser.append(f"[{timestamp}] {message}")

        # 自动滚动到底部
        self.textBrowser.verticalScrollBar().setValue(
            self.textBrowser.verticalScrollBar().maximum()
        )