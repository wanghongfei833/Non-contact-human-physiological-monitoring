#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Non-contact-human-physiological-monitoring 
@File    ：change_yaml.py
@IDE     ：PyCharm 
@Author  ：王洪飞
@Date    ：2025/12/25 15:18 
'''

import copy
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import shutil
import os

# 设置日志
logger = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QMessageBox, QFileDialog, QLabel, QScrollArea, QWidget
)
from PySide6.QtCore import Qt


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str = "./config.yaml"):
        self.config_path = Path(config_path)
        self.config_data: Optional[Dict[str, Any]] = None
        self.original_config: Optional[Dict[str, Any]] = None
        self.default_config_path = self.config_path.parent / "backups" / "config_default.yaml"

    def load(self) -> bool:
        """加载配置"""
        try:
            if not self.config_path.exists():
                logger.error(f"配置文件不存在: {self.config_path}")
                return False

            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config_data = yaml.safe_load(file)
                self.original_config = copy.deepcopy(self.config_data)

            logger.info(f"配置文件加载成功: {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False

    def save(self, config_content: str) -> bool:
        """保存配置"""
        try:
            # 验证YAML格式
            yaml.safe_load(config_content)

            # 创建备份
            self.create_backup()

            # 保存配置
            with open(self.config_path, 'w', encoding='utf-8') as file:
                file.write(config_content)

            # 重新加载配置
            self.load()

            logger.info("配置文件保存成功")
            return True
        except yaml.YAMLError as e:
            logger.error(f"YAML格式错误: {e}")
            raise ValueError(f"YAML格式错误: {e}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False

    def create_backup(self) -> str:
        """创建配置文件备份"""
        try:
            if not self.config_path.exists():
                return ""

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.config_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)

            backup_path = backup_dir / f"{self.config_path.stem}_backup_{timestamp}.yaml"
            shutil.copy2(self.config_path, backup_path)
            logger.info(f"配置文件备份创建成功: {backup_path}")

            return str(backup_path)

        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return ""

    def create_default_backup(self) -> bool:
        """创建默认配置备份（首次使用时调用）"""
        try:
            if not self.config_path.exists():
                logger.error(f"配置文件不存在，无法创建默认备份: {self.config_path}")
                return False

            # 创建备份目录
            self.default_config_path.parent.mkdir(exist_ok=True)

            # 如果默认备份已存在，不重复创建
            if self.default_config_path.exists():
                logger.info("默认备份已存在，跳过创建")
                return True

            # 创建默认备份
            shutil.copy2(self.config_path, self.default_config_path)
            logger.info(f"默认配置备份创建成功: {self.default_config_path}")

            return True

        except Exception as e:
            logger.error(f"创建默认备份失败: {e}")
            return False

    def restore_default(self) -> bool:
        """恢复默认配置"""
        try:
            if not self.default_config_path.exists():
                logger.error(f"默认备份文件不存在: {self.default_config_path}")
                return False

            # 创建当前配置的备份
            self.create_backup()

            # 恢复默认配置
            shutil.copy2(self.default_config_path, self.config_path)

            # 重新加载配置
            self.load()

            logger.info(f"配置从默认备份恢复成功: {self.default_config_path}")
            return True

        except Exception as e:
            logger.error(f"恢复默认备份失败: {e}")
            return False

    def get_config_content(self) -> str:
        """获取配置文件内容"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"读取配置文件内容失败: {e}")
            return ""

    def is_modified(self, current_content: str) -> bool:
        """检查配置是否被修改"""
        try:
            original_content = self.get_config_content()
            return original_content != current_content
        except Exception as e:
            logger.error(f"检查配置修改状态失败: {e}")
            return True

    def get_backup_files(self) -> list:
        """获取所有备份文件"""
        try:
            backup_dir = self.config_path.parent / "backups"
            if not backup_dir.exists():
                return []

            backup_files = []
            for file in backup_dir.glob(f"{self.config_path.stem}_backup_*.yaml"):
                backup_files.append({
                    'path': str(file),
                    'name': file.name,
                    'size': file.stat().st_size,
                    'modified': file.stat().st_mtime
                })

            # 按修改时间排序（最新的在前）
            backup_files.sort(key=lambda x: x['modified'], reverse=True)
            return backup_files

        except Exception as e:
            logger.error(f"获取备份文件失败: {e}")
            return []


class SimpleConfigEditor(QDialog):
    """简化的配置编辑器"""

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.original_content = ""
        self.is_content_modified = False

        self.setup_ui()
        self.load_config()
        self.apply_dark_style()

        # 首次使用时创建默认备份
        self.config_manager.create_default_backup()

    def setup_ui(self):
        self.setWindowTitle("配置文件编辑器")
        self.setGeometry(100, 100, 900, 700)

        layout = QVBoxLayout()

        # 标题和说明
        title_label = QLabel("YAML 配置文件编辑器")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)

        info_label = QLabel("直接编辑下方的YAML配置内容，修改后点击保存生效")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #AAAAAA; padding: 5px;")
        layout.addWidget(info_label)

        # 配置文件显示/编辑区域
        self.text_edit = QTextEdit()
        self.text_edit.setFontFamily("Courier New")
        self.text_edit.setFontPointSize(10)
        self.text_edit.textChanged.connect(self.on_content_changed)
        layout.addWidget(self.text_edit)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #00FF00; padding: 5px;")
        layout.addWidget(self.status_label)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.load_btn = QPushButton("重新加载")
        self.load_btn.clicked.connect(self.load_config)
        button_layout.addWidget(self.load_btn)

        self.restore_default_btn = QPushButton("恢复默认")
        self.restore_default_btn.clicked.connect(self.restore_default_config)
        self.restore_default_btn.setToolTip("恢复为首次打开时的配置")
        button_layout.addWidget(self.restore_default_btn)

        button_layout.addStretch()

        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)
        button_layout.addWidget(self.ok_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def apply_dark_style(self):
        """应用深色主题样式"""
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E2E;
                color: #FFFFFF;
            }
            QTextEdit {
                background-color: #2D2D44;
                color: #E0E0E0;
                border: 1px solid #404040;
                border-radius: 4px;
                font-family: "Courier New", monospace;
                selection-background-color: #2E8B57;
            }
            QPushButton {
                background-color: #2D2D44;
                color: #E0E0E0;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #3D3D54;
                border: 1px solid #505050;
            }
            QPushButton:disabled {
                background-color: #1A1A2E;
                color: #606060;
            }
            QPushButton:pressed {
                background-color: #2E8B57;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 12px;
            }
        """)

    def load_config(self):
        """加载配置文件"""
        try:
            content = self.config_manager.get_config_content()
            self.text_edit.setPlainText(content)
            self.original_content = content
            self.is_content_modified = False
            self.update_ui_state()
            self.set_status("配置加载成功", "green")
        except Exception as e:
            self.set_status(f"加载配置失败: {str(e)}", "red")
            QMessageBox.critical(self, "错误", f"加载配置文件失败:\n{str(e)}")

    def save_config(self):
        """保存配置文件"""
        try:
            content = self.text_edit.toPlainText()

            # 验证并保存配置
            self.config_manager.save(content)

            self.original_content = content
            self.is_content_modified = False
            self.update_ui_state()
            self.set_status("配置保存成功", "green")

            QMessageBox.information(self, "成功", "配置文件保存成功！")

        except ValueError as e:
            self.set_status(f"保存失败: {str(e)}", "red")
            QMessageBox.critical(self, "YAML格式错误", f"YAML格式错误:\n{str(e)}")
        except Exception as e:
            self.set_status(f"保存失败: {str(e)}", "red")
            QMessageBox.critical(self, "错误", f"保存配置文件失败:\n{str(e)}")

    def restore_default_config(self):
        """恢复默认配置"""
        try:
            reply = QMessageBox.question(
                self,
                "确认恢复",
                "确定要恢复为默认配置吗？当前修改将丢失。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                if self.config_manager.restore_default():
                    self.load_config()
                    self.set_status("已恢复默认配置", "green")
                    QMessageBox.information(self, "成功", "配置已恢复为默认值！")
                else:
                    self.set_status("恢复默认配置失败", "red")
            else:
                self.set_status("取消恢复操作", "yellow")

        except Exception as e:
            self.set_status(f"恢复默认配置失败: {str(e)}", "red")
            QMessageBox.critical(self, "错误", f"恢复默认配置失败:\n{str(e)}")

    def on_content_changed(self):
        """内容变化处理"""
        current_content = self.text_edit.toPlainText()
        self.is_content_modified = (current_content != self.original_content)
        self.update_ui_state()

        if self.is_content_modified:
            self.set_status("内容已修改", "yellow")
        else:
            self.set_status("就绪", "green")

    def update_ui_state(self):
        """更新UI状态"""
        self.save_btn.setEnabled(self.is_content_modified)
        self.ok_btn.setEnabled(self.is_content_modified)

    def set_status(self, message: str, color: str = "white"):
        """设置状态栏消息"""
        color_map = {
            "green": "#00FF00",
            "red": "#FF0000",
            "yellow": "#FFFF00",
            "white": "#FFFFFF"
        }
        color_hex = color_map.get(color, "#FFFFFF")
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color_hex}; padding: 5px;")

    def accept(self):
        """确定按钮点击"""
        if self.is_content_modified:
            reply = QMessageBox.question(
                self, "确认",
                "配置已修改，是否保存？",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.save_config()
                super().accept()
            elif reply == QMessageBox.StandardButton.No:
                super().accept()
            else:
                return  # 取消操作
        else:
            super().accept()


# 为了保持与main.py的兼容性，保留原有的ConfigEditorDialog类名
# 但实际使用SimpleConfigEditor的功能
class ConfigEditorDialog(SimpleConfigEditor):
    """为了保持兼容性，将SimpleConfigEditor重命名为ConfigEditorDialog"""
    pass