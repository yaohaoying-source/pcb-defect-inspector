"""
设置对话框 - 配置服务器、API Key 和检测参数
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QPushButton,
    QDialogButtonBox
)
from PyQt5.QtCore import Qt

from utils.config import AppConfig


class SettingsDialog(QDialog):
    """应用设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("应用设置")
        self.resize(450, 350)
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ---- 服务器配置 ----
        server_group = QGroupBox("服务器配置")
        server_layout = QFormLayout(server_group)
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("例如: 127.0.0.1")
        server_layout.addRow("服务器 IP:", self.host_edit)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        server_layout.addRow("图片上传端口:", self.port_spin)

        self.web_port_spin = QSpinBox()
        self.web_port_spin.setRange(1, 65535)
        server_layout.addRow("模型下载端口:", self.web_port_spin)

        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("/userdata/Model")
        server_layout.addRow("模型保存目录:", self.model_dir_edit)
        layout.addWidget(server_group)

        # ---- DeepSeek 配置 ----
        deepseek_group = QGroupBox("DeepSeek 配置")
        ds_layout = QFormLayout(deepseek_group)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-...")
        ds_layout.addRow("API Key:", self.api_key_edit)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-chat", "deepseek-reasoner"])
        ds_layout.addRow("模型:", self.model_combo)
        layout.addWidget(deepseek_group)

        # ---- 检测参数 ----
        detect_group = QGroupBox("检测参数")
        detect_layout = QFormLayout(detect_group)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        detect_layout.addRow("置信度阈值:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        detect_layout.addRow("IOU 阈值:", self.iou_spin)
        layout.addWidget(detect_group)

        # ---- 按钮 ----
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_settings(self):
        """从配置加载当前值"""
        self.host_edit.setText(AppConfig.get_server_host())
        self.port_spin.setValue(AppConfig.get_server_port())
        self.web_port_spin.setValue(AppConfig.get_server_web_port())
        self.model_dir_edit.setText(AppConfig.get_model_download_dir())
        self.api_key_edit.setText(AppConfig.get_deepseek_api_key())

        model = AppConfig.get_deepseek_model()
        idx = self.model_combo.findText(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.conf_spin.setValue(AppConfig.get_confidence())
        self.iou_spin.setValue(AppConfig.get_iou())

    def _save_settings(self):
        """保存配置"""
        AppConfig.set_server_host(self.host_edit.text().strip())
        AppConfig.set_server_port(self.port_spin.value())
        AppConfig.set_server_web_port(self.web_port_spin.value())
        AppConfig.set_model_download_dir(self.model_dir_edit.text().strip() or "/userdata/Model")
        AppConfig.set_deepseek_api_key(self.api_key_edit.text().strip())
        AppConfig.set_deepseek_model(self.model_combo.currentText())
        AppConfig.set_confidence(self.conf_spin.value())
        AppConfig.set_iou(self.iou_spin.value())
        self.accept()

