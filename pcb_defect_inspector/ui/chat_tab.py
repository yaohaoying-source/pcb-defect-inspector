"""
DeepSeek 聊天标签页 - API 对话界面
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextBrowser, QTextEdit, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from utils.config import AppConfig
from deepseek.api_client import DeepSeekClient


class ChatWorker(QThread):
    """异步 DeepSeek API 调用工作器"""
    signal_response = pyqtSignal(str, str)  # (role, content) 流式片段
    signal_done = pyqtSignal(str)           # 完整回复
    signal_error = pyqtSignal(str)          # 错误

    def __init__(self, api_key, model, messages):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.messages = messages

    def run(self):
        try:
            client = DeepSeekClient(self.api_key, self.model)
            reply = client.send_message(self.messages)
            self.signal_done.emit(reply)
        except Exception as e:
            self.signal_error.emit(str(e))


class ChatTab(QWidget):
    """DeepSeek 聊天标签页"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.messages = []          # 对话历史 [{"role", "content"}, ...]
        self.chat_worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ---- API Key 状态栏 ----
        key_layout = QHBoxLayout()
        self.lbl_api_status = QLabel("API Key: 未设置")
        self.lbl_api_status.setStyleSheet("color: #e74c3c;")
        key_layout.addWidget(self.lbl_api_status)
        key_layout.addStretch()
        self.btn_set_key = QPushButton("设置 API Key")
        self.btn_set_key.clicked.connect(self._configure_api_key)
        key_layout.addWidget(self.btn_set_key)
        layout.addLayout(key_layout)

        # ---- 聊天记录显示 ----
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.setMinimumHeight(400)
        layout.addWidget(self.chat_display, stretch=1)

        # ---- 输入区域 ----
        input_layout = QHBoxLayout()

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("输入消息给 DeepSeek... (Ctrl+Enter 发送)")
        self.input_edit.setMaximumHeight(80)
        self.input_edit.setAcceptRichText(False)
        input_layout.addWidget(self.input_edit, stretch=1)

        self.btn_send = QPushButton("发送")
        self.btn_send.clicked.connect(self._send_message)
        self.btn_send.setEnabled(False)
        self.btn_send.setMinimumWidth(80)
        input_layout.addWidget(self.btn_send)

        self.btn_clear_chat = QPushButton("清空对话")
        self.btn_clear_chat.clicked.connect(self._clear_chat)
        input_layout.addWidget(self.btn_clear_chat)

        layout.addLayout(input_layout)

        # 检查是否已有 API Key
        self._refresh_api_status()

    def _refresh_api_status(self):
        """刷新 API Key 状态显示"""
        api_key = AppConfig.get_deepseek_api_key()
        if api_key:
            masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 12 else "已设置"
            self.lbl_api_status.setText(f"API Key: {masked}")
            self.lbl_api_status.setStyleSheet("color: #27ae60;")
            self.btn_send.setEnabled(True)
        else:
            self.lbl_api_status.setText("API Key: 未设置")
            self.lbl_api_status.setStyleSheet("color: #e74c3c;")
            self.btn_send.setEnabled(False)

    def _configure_api_key(self):
        """打开设置对话框配置 API Key"""
        from .settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        if dialog.exec_():
            self._refresh_api_status()

    def _send_message(self):
        """发送消息到 DeepSeek"""
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        api_key = AppConfig.get_deepseek_api_key()
        if not api_key:
            QMessageBox.warning(self, "提示", "请先在设置中配置 DeepSeek API Key")
            return

        # 添加用户消息
        self.messages.append({"role": "user", "content": text})
        self._append_message("user", text)
        self.input_edit.clear()

        # 禁用发送按钮
        self.btn_send.setEnabled(False)
        self.btn_send.setText("请求中...")

        # 启动异步请求
        model = AppConfig.get_deepseek_model()
        self.chat_worker = ChatWorker(api_key, model, self.messages)
        self.chat_worker.signal_done.connect(self._on_reply_done)
        self.chat_worker.signal_error.connect(self._on_reply_error)
        self.chat_worker.start()

    def _on_reply_done(self, reply):
        """收到 DeepSeek 回复"""
        self.messages.append({"role": "assistant", "content": reply})
        self._append_message("assistant", reply)
        self.btn_send.setEnabled(True)
        self.btn_send.setText("发送")

    def _on_reply_error(self, error_msg):
        """DeepSeek 回复错误"""
        self._append_message("system", f"错误: {error_msg}")
        self.btn_send.setEnabled(True)
        self.btn_send.setText("发送")

    def _append_message(self, role, content):
        """在聊天显示区追加一条消息"""
        if role == "user":
            html = (
                f'<div style="margin: 8px 0; text-align: right;">'
                f'<div style="display: inline-block; background: #007aff; color: white; '
                f'padding: 8px 14px; border-radius: 12px 12px 4px 12px; '
                f'max-width: 80%; text-align: left;">'
                f'{content}</div></div>'
            )
        elif role == "assistant":
            html = (
                f'<div style="margin: 8px 0; text-align: left;">'
                f'<div style="display: inline-block; background: #e9e9eb; color: #1c1c1e; '
                f'padding: 8px 14px; border-radius: 12px 12px 12px 4px; '
                f'max-width: 80%; text-align: left;">'
                f'{content}</div></div>'
            )
        else:
            html = (
                f'<div style="margin: 8px 0; text-align: center; color: #e74c3c;">'
                f'{content}</div>'
            )

        self.chat_display.append(html)
        # 滚动到底部
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_chat(self):
        """清空对话历史"""
        self.messages = []
        self.chat_display.clear()
