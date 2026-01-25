from PySide6.QtGui import QTextCursor, QTextCharFormat, QFont, QColor
from datetime import datetime


class LOG(object):
    def __init__(self, text_browser=None,logger_write=None):
        self.text_browser = text_browser
        self.logger_write = logger_write

    def log_info_enhanced(self, text, level="INFO", color=None, font_size=10, timestamp=True, text_browser=None):
        """
        增强版日志输出函数，默认使用 self.textBrowser

        参数:
            text (str): 要输出的文本内容
            level (str): 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)，默认为INFO
            color (str): 自定义颜色，如果为None则根据级别自动选择
                level_colors = {
                "DEBUG": "blue",
                "INFO": "#F0F8FF",
                "WARNING": "orange",
                "ERROR": "red",
                "CRITICAL": "#FF0000"  # 亮红色
                }


            font_size (int): 字体大小，默认为10
            timestamp (bool): 是否添加时间戳，默认为True
            text_browser (QTextBrowser): 目标文本浏览器控件，默认为self.textBrowser

        """
        try:
            # 确定目标文本浏览器
            if text_browser is None:
                text_browser = self.text_browser

            # 定义日志级别对应的颜色
            level_colors = {
                "DEBUG": "blue",
                "INFO": "#F0F8FF",
                "WARNING": "orange",
                "ERROR": "red",
                "CRITICAL": "#FF0000"  # 亮红色
            }

            # 获取当前时间
            if timestamp:
                timestamp_str = datetime.now().strftime("[%H:%M:%S] ")
            else:
                timestamp_str = ""

            # 确定颜色
            if color is None:
                color = level_colors.get(level.upper(), 'blue')

            # 创建格式化的日志行
            level_str = f"[{level}] " if level else ""
            full_text = f"{timestamp_str}{level_str}{text}"

            # 保存当前光标位置
            cursor = text_browser.textCursor()

            # 创建文本格式对象
            char_format = QTextCharFormat()

            # 设置字体
            font = QFont()
            font.setPointSize(font_size)
            char_format.setFont(font)

            # 设置文本颜色
            qcolor = QColor()
            qcolor.setNamedColor(color)
            char_format.setForeground(qcolor)

            # 移动到文档末尾
            cursor.movePosition(QTextCursor.End)

            # 应用格式并插入文本
            cursor.insertText(full_text, char_format)

            # 添加换行符
            cursor.insertText("\n")

            # 更新文本浏览器
            text_browser.setTextCursor(cursor)
            text_browser.ensureCursorVisible()

            # 自动滚动到底部
            text_browser.verticalScrollBar().setValue(text_browser.verticalScrollBar().maximum())
            self.logger_write.info(full_text)
        except Exception as e:
            print(text)
