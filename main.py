import atexit
import faulthandler
import os.path
import traceback
import warnings
from datetime import datetime

import yaml as pyyaml
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QMouseEvent, QColor, QPalette
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog

from utils.base_utils import *
from utils.change_yaml import ConfigEditorDialog, SimpleConfigEditor, ConfigManager
from utils.surface_utils import SurfaceUtils

# ==================== 第一步：基础环境配置 ====================
# 设置环境变量，防止CUDA相关问题
os.environ['ORT_CUDA_GRAPH_CAPTURE_MODE'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行，便于调试
os.environ['PYTHONUNBUFFERED'] = '1'  # 无缓冲输出
os.environ['QT_DEBUG_PLUGINS'] = '0'  # 禁用Qt插件调试
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'  # 禁用dynamo
os.environ['TORCH_COMPILE_DEBUG'] = '0'  # 关闭编译调试
os.environ['ORT_LOG_LEVEL'] = '0'  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
# 启用故障处理器，捕获段错误
faulthandler.enable()
# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))




try:
    with open('./config.yaml', 'r', encoding='utf-8') as file:
        config_data = pyyaml.safe_load(file)
except Exception as e:
    raise


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_root = config_data['EXPORT_INFO']['LOG_PATH']
log_file = f"{log_root}/app_{timestamp}.log"
save_dir = config_data['EXPORT_INFO']['SAVE_PATH']


# 创建日志目录

os.makedirs(log_root, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

"""配置详细的日志记录"""

logger = setup_logging(log_file)


# ==================== 第二步：异常处理 ====================
def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """处理未捕获的异常"""
    if issubclass(exc_type, KeyboardInterrupt):
        # 忽略键盘中断
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("=" * 60)
    logger.error("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))
    logger.error("=" * 60)

    # 尝试将错误信息写入文件
    try:
        with open("logs/crash_report.txt", "w", encoding='utf-8') as f:
            f.write(f"崩溃时间: {datetime.now()}\n")
            f.write(f"异常类型: {exc_type.__name__}\n")
            f.write(f"异常信息: {exc_value}\n")
            f.write("\n堆栈跟踪:\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    except Exception as e:
        logger.error(f"写入崩溃报告失败: {e}")


sys.excepthook = handle_uncaught_exception

# ==================== 第三步：版本兼容性 ====================
# 处理Python版本兼容性问题
if sys.version_info < (3, 11):
    import typing


    class SelfMeta(type):
        def __getitem__(self, item):
            return typing.Any


    class Self(metaclass=SelfMeta):
        pass


    typing.Self = Self

# 过滤警告
warnings.filterwarnings('ignore')


# ==================== 第四步：导入检查 ====================
def check_imports():
    """检查关键模块导入"""
    required_modules = [
        'PySide6',
        'yaml',
        'onnxruntime',
        'torch',
        'numpy',
        'cv2',
    ]

    logger.info("检查模块导入...")
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError as _e:
            logger.error(f"✗ {module}: {_e}")
            if module in ['PySide6', 'yaml']:
                raise


check_imports()

# ==================== 第五步：主程序 ====================


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # # 添加这3行代码
        # self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框窗口
        # self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        # self.setStyleSheet("""
        #             QMainWindow {
        #                 background-color: #1E1E2E;
        #                 border: 1px solid #2E8B57;
        #                 border-radius: 1px;
        #             }
        #         """)
        #
        # 设置窗口背景色
        self.setStyleSheet("""
            /* 主窗口样式 */
            QMainWindow {
                background-color: #1E1E2E;
            }

            /* 设置一个深色的边框，尽量接近系统标题栏颜色 */
            QMainWindow::separator {
                background-color: #1E1E2E;
            }
        """)

        # 设置窗口调色板
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#161618"))
        self.setPalette(palette)

        self.config_data = config_data
        self.ui = SurfaceUtils(qmain_window=self, config_data=self.config_data,logger_write = logger)
        self.ui.star_carm()
        # 设置窗口关闭确认
        self.close_confirmed = False
        # 创建定时器检查线程状态
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.check_system_status)
        self.check_timer.start(5000)  # 每5秒检查一次
        logger.info("MainWindow初始化完成")
        # ===================== 链接按钮 ==========================
        # --------------------- 链接历史记录 -----------------------
        self.ui.history_button.clicked.connect(self._load_history)
        # --------------------- 链接日志查看按钮 --------------------
        self.ui.log_button.clicked.connect(self._load_log)
        # --------------------- 链接配置文件修改 --------------------
        self.ui.config_button.clicked.connect(self.change_config_yaml)
        # --------------------- 链接退出按钮 --------------------
        self.ui.exit_button.clicked.connect(self._exit)


    @staticmethod
    def _load_log():
        open_folder(log_root)


    def _load_history(self):
        history_path = self.config_data['EXPORT_INFO']['SAVE_PATH']
        if os.path.exists(history_path) and os.path.isdir(history_path):
            open_folder(history_path)

    def closeEvent(self, event):
        """重写关闭事件，显示确认对话框"""
        logger.info("收到关闭事件")

        if not self.close_confirmed:
            reply = QMessageBox.question(
                self,
                "视频流未关闭",
                "确定要退出视频流，并且关闭程序吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                logger.info("用户确认退出")
                self.close_confirmed = True
                self.cleanup_before_exit()
                event.accept()  # 接受关闭事件，程序退出
            else:
                logger.info("用户取消退出")
                event.ignore()  # 忽略关闭事件，程序继续运行
        else:
            logger.info("已确认退出，执行清理")
            self.cleanup_before_exit()
            event.accept()
    def change_config_yaml(self):
        """
        打开配置编辑弹窗
        """
        try:
            # 创建配置编辑对话框
            # dialog = ConfigEditorDialog('./config.yaml', self)
            config_namger = ConfigManager('./config.yaml')
            dialog = ConfigEditorDialog(config_namger, self)
            if dialog.exec() == QDialog.Accepted:
                QMessageBox.information(self, "成功", "配置已保存，请重启程序使配置生效！")
                self.ui.logger.log_info_enhanced("成功","重启后生效")
            else:
                # 用户取消了编辑
                logger.info("用户取消了配置编辑")

        except Exception as _e:
            logger.error(f"打开配置编辑器失败: {_e}")
            print(_e)
            QMessageBox.critical(self, "错误", f"打开配置编辑器失败:\n{str(_e)}")

    @staticmethod
    def check_system_status():
        """定期检查系统状态"""
        try:
            import threading
            active_threads = threading.enumerate()
            logger.debug(f"活动线程数: {len(active_threads)}")

            # 检查内存使用
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_usage = process.memory_info().rss / 1024 / 1024
            logger.debug(f"内存使用: {mem_usage:.1f} MB")

        except Exception as _e:
            logger.debug(f"状态检查失败: {_e}")

    def _exit(self):
        """重写关闭事件，显示确认对话框"""
        logger.info("收到关闭事件")

        if not self.close_confirmed:
            reply = QMessageBox.question(
                self,
                "视频流未关闭",
                "确定要退出视频流，并且关闭程序吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                logger.info("用户确认退出")
                self.close_confirmed = True
                self.cleanup_before_exit()
                QApplication.quit()
            else:
                logger.info("用户取消退出")
        else:
            logger.info("已确认退出，执行清理")
            self.cleanup_before_exit()
            QApplication.quit()

    def cleanup_before_exit(self):
        """程序退出前的清理工作"""
        logger.info("开始清理资源...")

        try:
            # 停止检查定时器
            if hasattr(self, 'check_timer'):
                self.check_timer.stop()

            # 清理UI资源
            if hasattr(self, 'ui') and hasattr(self.ui, 'close_carm'):
                logger.info("关闭摄像头资源...")
                if self.ui is not None:
                    self.ui.close_carm()

            # 清理可能存在的其他资源
            import gc
            gc.collect()

            logger.info("资源清理完成")

        except Exception as _e:
            logger.error(f"清理资源时出错: {_e}")

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标事件，用于调试"""
        logger.debug(f"鼠标点击: {event.pos()}")
        super().mousePressEvent(event)


# ==================== 第六步：应用程序类 ====================
class SafeApplication(QApplication):
    """安全的应用程序类，添加额外的事件处理"""

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("应用程序创建")

        # 连接aboutToQuit信号
        self.aboutToQuit.connect(self.on_about_to_quit)

        # 设置应用程序名称
        self.setApplicationName("Non-contact Physiological Monitoring")
        self.setApplicationVersion("1.0.0")

    @staticmethod
    def on_about_to_quit():
        """应用程序即将退出"""
        logger.info("应用程序即将退出")

    def notify(self, receiver, event):
        """重写notify方法，捕获所有事件"""
        try:
            return super().notify(receiver, event)
        except Exception as e:
            logger.error(f"事件处理异常: {e}", exc_info=True)
            return False


# ==================== 第七步：程序入口 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("程序启动")
    logger.info("=" * 60)

    app = None
    window = None

    try:
        # 启用高DPI支持
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        # 创建应用程序
        logger.info("创建QApplication...")
        app = SafeApplication([])

        # 设置应用程序图标（可选）
        # from PySide6.QtGui import QIcon
        # app.setWindowIcon(QIcon("icon.png"))

        # 加载配置文件
        config_yaml_path = "./config.yaml"
        if not os.path.exists(config_yaml_path):
            config_yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        if not os.path.exists(config_yaml_path):
            logger.error(f"配置文件不存在: {config_yaml_path}")
            raise FileNotFoundError(f"配置文件不存在: {config_yaml_path}")

        # 创建主窗口
        logger.info("创建主窗口...")
        window = MainWindow()

        # 注册退出处理函数
        atexit.register(lambda: logger.info("程序退出"))

        # 显示窗口
        logger.info("显示主窗口")
        window.show()

        # 运行应用程序
        logger.info("进入事件循环")
        exit_code = app.exec()
        logger.info(f"事件循环结束，退出码: {exit_code}")

        return exit_code

    except Exception as e:
        logger.critical(f"程序启动失败: {e}", exc_info=True)

        # 显示错误对话框
        if app and QApplication.instance():
            QMessageBox.critical(
                None,
                "程序启动失败",
                f"程序启动失败:\n{str(e)}\n\n"
                f"请查看日志文件获取详细信息。"
            )
        return 1

    finally:
        logger.info("程序结束")
        if app:
            app.quit()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"程序崩溃: {e}", exc_info=True)
        sys.exit(1)