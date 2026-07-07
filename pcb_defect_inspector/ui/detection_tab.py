"""
检测标签页 - 图片加载、YOLO 检测、结果展示和上传
"""
import os
import html
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QScrollArea, QProgressBar, QSplitter, QFrame, QTextBrowser, QTextEdit,
    QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor

from utils.config import AppConfig
from utils.image_loader import load_bgr_with_orientation
from utils.model_downloader import download_remote_model, list_remote_models
from utils.result_paths import build_result_image_path
from detection.rknn_detector import RknnDetector
from vl.qwen_vl_client import QwenVLSession
from vl.qwen_prompt import build_qwen_prompt


class BackgroundImageLabel(QLabel):
    """Preview label that scales the background fully before an image is loaded."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._background = QPixmap()

    def set_background_image(self, path):
        self._background = QPixmap(path)
        self.update()

    def paintEvent(self, event):
        pixmap = self.pixmap()
        if (pixmap is None or pixmap.isNull()) and not self._background.isNull():
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#ffffff"))
            target = self.rect().adjusted(8, 8, -8, -8)
            scaled = self._background.scaled(
                target.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            x = target.x() + (target.width() - scaled.width()) // 2
            y = target.y() + (target.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()
        super().paintEvent(event)


class DetectionWorker(QThread):
    """异步检测工作器，防止 UI 阻塞"""
    signal_done = pyqtSignal(object)   # 检测结果图像 (np.ndarray)
    signal_error = pyqtSignal(str)     # 错误消息
    signal_progress = pyqtSignal(str)  # 状态文本

    def __init__(self, detector, image, conf, iou):
        super().__init__()
        self.detector = detector
        self.image = image.copy()
        self.conf = conf
        self.iou = iou

    def run(self):
        try:
            result_img, detections = self.detector.detect(
                self.image, conf=self.conf, iou=self.iou
            )
            self.signal_done.emit((result_img, detections))
        except Exception as e:
            self.signal_error.emit(str(e))


class UploadWorker(QThread):
    """异步上传工作器"""
    signal_progress = pyqtSignal(int)      # 0-100%
    signal_finished = pyqtSignal(bool, str)  # 成功/失败 + 消息

    def __init__(self, image_path, host, port):
        super().__init__()
        self.image_path = image_path
        self.host = host
        self.port = port

    def run(self):
        try:
            from upload.uploader import upload_file
            upload_file(
                self.image_path, self.host, self.port,
                progress_callback=self.signal_progress.emit
            )
            self.signal_finished.emit(True, "上传成功")
        except Exception as e:
            self.signal_finished.emit(False, f"上传失败: {e}")


class ModelListWorker(QThread):
    """异步读取服务器模型列表"""
    signal_done = pyqtSignal(list)
    signal_error = pyqtSignal(str)

    def __init__(self, host, web_port):
        super().__init__()
        self.host = host
        self.web_port = web_port

    def run(self):
        try:
            self.signal_done.emit(list_remote_models(self.host, self.web_port))
        except Exception as e:
            self.signal_error.emit(str(e))


class ModelDownloadWorker(QThread):
    """异步从服务器下载 RKNN 模型到板卡"""
    signal_progress = pyqtSignal(int)
    signal_finished = pyqtSignal(bool, str, str)

    def __init__(self, host, web_port, filename, target_dir):
        super().__init__()
        self.host = host
        self.web_port = web_port
        self.filename = filename
        self.target_dir = target_dir

    def run(self):
        try:
            path = download_remote_model(
                self.host,
                self.filename,
                self.target_dir,
                self.web_port,
                progress_callback=self.signal_progress.emit,
            )
            self.signal_finished.emit(True, "模型下载完成", path)
        except Exception as e:
            self.signal_finished.emit(False, f"模型下载失败: {e}", "")


class QwenAnalysisWorker(QThread):
    """异步调用本地 Qwen3-VL demo，避免阻塞 UI"""
    signal_done = pyqtSignal(str)
    signal_error = pyqtSignal(str)

    def __init__(self, session, question):
        super().__init__()
        self.session = session
        self.question = question

    def run(self):
        try:
            reply = self.session.ask(self.question)
            self.signal_done.emit(reply)
        except Exception as e:
            self.signal_error.emit(str(e))


class DetectionTab(QWidget):
    """目标检测标签页 - 加载模型、选择图片、检测、展示、上传"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        self.detector = RknnDetector()
        self.current_image = None       # 原始图像 (np.ndarray, RGB)
        self.current_result = None      # 检测结果图像 (np.ndarray, RGB)
        self.current_result_path = None # 结果图片临时路径
        self.detection_worker = None
        self.upload_worker = None
        self.model_list_worker = None
        self.model_download_worker = None
        self.qwen_worker = None
        self.qwen_session = None
        self.qwen_session_image_path = ""
        self.current_detections = []
        self.qwen_history = []
        self._pending_qwen_question = ""

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ---- 工具栏 ----
        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.open_image)
        self.btn_detect = QPushButton("检测")
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_detect.setEnabled(False)
        self.btn_upload = QPushButton("上传到服务器")
        self.btn_upload.clicked.connect(self.start_upload)
        self.btn_upload.setEnabled(False)
        self.btn_clear = QPushButton("清除")
        self.btn_clear.clicked.connect(self.clear_results)

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_detect)
        toolbar.addWidget(self.btn_upload)
        toolbar.addWidget(self.btn_clear)
        toolbar.addStretch()

        # ---- 模型信息 ----
        self.lbl_model = QLabel("模型: 未加载")
        self.lbl_model.setStyleSheet("color: #888;")
        toolbar.addWidget(self.lbl_model)

        layout.addLayout(toolbar)

        # ---- 分割视图: 左侧图片预览 + 右侧控制面板 ----
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：图片预览
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets",
        )
        preview_bg = os.path.join(assets_dir, "app_background_faded.jpg")
        self.image_label = BackgroundImageLabel("请点击「打开图片」加载图片")
        self.image_label.setObjectName("ImagePreviewLabel")
        self.image_label.set_background_image(preview_bg)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 320)
        self.image_label.setStyleSheet("""
            QLabel#ImagePreviewLabel {
                border: 1px solid #c8d3dc;
                color: #244860;
            }
        """)
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        preview_layout.addWidget(scroll)

        # 检测信息
        info_layout = QHBoxLayout()
        self.lbl_detection_info = QLabel("")
        info_layout.addWidget(self.lbl_detection_info)
        info_layout.addStretch()
        preview_layout.addLayout(info_layout)

        splitter.addWidget(preview_widget)

        # 右侧：控制面板
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)
        panel_layout.setContentsMargins(6, 6, 6, 6)

        # 模型加载组
        model_group = QGroupBox("模型")
        model_layout = QVBoxLayout(model_group)
        self.btn_load_model = QPushButton("加载 .rknn 模型")
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        model_layout.addWidget(self.btn_load_model)

        self.remote_model_combo = QComboBox()
        self.remote_model_combo.setMinimumContentsLength(18)
        self.remote_model_combo.addItem("点击刷新服务器模型")
        model_layout.addWidget(self.remote_model_combo)

        model_btn_layout = QHBoxLayout()
        self.btn_refresh_models = QPushButton("刷新服务器模型")
        self.btn_refresh_models.clicked.connect(self.refresh_remote_models)
        model_btn_layout.addWidget(self.btn_refresh_models)

        self.btn_download_model = QPushButton("下载到板卡")
        self.btn_download_model.clicked.connect(self.download_selected_model)
        self.btn_download_model.setEnabled(False)
        model_btn_layout.addWidget(self.btn_download_model)
        model_layout.addLayout(model_btn_layout)

        self.model_download_progress = QProgressBar()
        self.model_download_progress.setVisible(False)
        self.model_download_progress.setValue(0)
        model_layout.addWidget(self.model_download_progress)

        self.lbl_model_path = QLabel("")
        self.lbl_model_path.setWordWrap(True)
        self.lbl_model_path.setStyleSheet("color: #666; font-size: 11px;")
        model_layout.addWidget(self.lbl_model_path)
        panel_layout.addWidget(model_group)

        # 检测参数组
        param_group = QGroupBox("检测参数")
        param_layout = QFormLayout(param_group)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(AppConfig.get_confidence())
        param_layout.addRow("置信度:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(AppConfig.get_iou())
        param_layout.addRow("IOU:", self.iou_spin)

        panel_layout.addWidget(param_group)

        # 上传状态
        upload_group = QGroupBox("上传")
        upload_layout = QVBoxLayout(upload_group)
        self.upload_progress = QProgressBar()
        self.upload_progress.setVisible(False)
        self.upload_progress.setValue(0)
        upload_layout.addWidget(self.upload_progress)
        self.lbl_upload_status = QLabel("")
        upload_layout.addWidget(self.lbl_upload_status)
        panel_layout.addWidget(upload_group)

        # Qwen3-VL 问答
        qwen_group = QGroupBox("Qwen3-VL 问答")
        qwen_layout = QVBoxLayout(qwen_group)
        self.lbl_qwen_dir = QLabel(AppConfig.get_qwen_vl_dir())
        self.lbl_qwen_dir.setWordWrap(True)
        self.lbl_qwen_dir.setStyleSheet("color: #666; font-size: 11px;")
        qwen_layout.addWidget(self.lbl_qwen_dir)

        qwen_btn_layout = QHBoxLayout()
        self.btn_select_qwen_dir = QPushButton("选择目录")
        self.btn_select_qwen_dir.clicked.connect(self.select_qwen_dir)
        qwen_btn_layout.addWidget(self.btn_select_qwen_dir)

        self.btn_qwen_clear = QPushButton("清空对话")
        self.btn_qwen_clear.clicked.connect(self.clear_qwen_chat)
        qwen_btn_layout.addWidget(self.btn_qwen_clear)
        qwen_layout.addLayout(qwen_btn_layout)

        self.qwen_output = QTextBrowser()
        self.qwen_output.setMinimumHeight(140)
        qwen_layout.addWidget(self.qwen_output)

        self.qwen_input = QTextEdit()
        self.qwen_input.setMaximumHeight(58)
        self.qwen_input.setAcceptRichText(False)
        self.qwen_input.setPlaceholderText("检测完成后输入问题，例如：这种缺陷怎么改进工艺？")
        qwen_layout.addWidget(self.qwen_input)

        qwen_send_layout = QHBoxLayout()
        self.btn_qwen_suggest = QPushButton("默认分析")
        self.btn_qwen_suggest.clicked.connect(self.ask_default_qwen_question)
        self.btn_qwen_suggest.setEnabled(False)
        qwen_send_layout.addWidget(self.btn_qwen_suggest)

        self.btn_qwen_send = QPushButton("发送问题")
        self.btn_qwen_send.clicked.connect(self.start_qwen_analysis)
        self.btn_qwen_send.setEnabled(False)
        qwen_send_layout.addWidget(self.btn_qwen_send)
        qwen_layout.addLayout(qwen_send_layout)

        panel_layout.addWidget(qwen_group)

        panel_layout.addStretch()

        panel_scroll = QScrollArea()
        panel_scroll.setWidget(panel_widget)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setMaximumWidth(300)
        panel_scroll.setMinimumWidth(250)
        splitter.addWidget(panel_scroll)

        layout.addWidget(splitter, stretch=1)

        # ---- 状态 ----
        self.lbl_status = QLabel("就绪 - 请先加载模型，然后打开图片进行检测")
        self.lbl_status.setStyleSheet("color: #666; padding: 4px;")
        layout.addWidget(self.lbl_status)

    # ========== 公共方法 ==========

    def open_image(self):
        """打开图片文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"
        )
        if not path:
            return

        self._load_image_from_path(path)

    def _load_image_from_path(self, path):
        """从路径加载图片并显示"""
        self._close_qwen_session()
        img = load_bgr_with_orientation(path)
        if img is None:
            QMessageBox.warning(self, "错误", "无法加载图片，请检查文件格式")
            return

        self.current_image_path = path
        self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._display_image(self.current_image)
        self.btn_detect.setEnabled(self.detector.is_loaded())
        self.btn_upload.setEnabled(False)
        self.btn_qwen_send.setEnabled(False)
        self.btn_qwen_suggest.setEnabled(False)
        self.current_detections = []
        self.clear_qwen_chat()
        self.lbl_detection_info.setText("")
        self.lbl_status.setText(f"已加载图片: {os.path.basename(path)}")

        if hasattr(self, 'lbl_upload_status'):
            self.lbl_upload_status.setText("")

    def load_model_dialog(self):
        """打开模型文件对话框"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", AppConfig.get_last_model_path(),
            "RKNN 模型 (*.rknn);;所有文件 (*.*)"
        )
        if not path:
            return

        self._load_model(path)

    def refresh_remote_models(self):
        """从服务器读取可下载的 RKNN 模型列表"""
        host = AppConfig.get_server_host()
        web_port = AppConfig.get_server_web_port()
        self.btn_refresh_models.setEnabled(False)
        self.btn_download_model.setEnabled(False)
        self.remote_model_combo.clear()
        self.remote_model_combo.addItem("正在读取模型列表...")
        self.lbl_status.setText(f"读取服务器模型列表: {host}:{web_port}...")

        self.model_list_worker = ModelListWorker(host, web_port)
        self.model_list_worker.signal_done.connect(self._on_remote_models_loaded)
        self.model_list_worker.signal_error.connect(self._on_remote_models_error)
        self.model_list_worker.start()

    def download_selected_model(self):
        """下载服务器上的模型到板卡本地目录"""
        model = self.remote_model_combo.currentData()
        if not model:
            QMessageBox.warning(self, "提示", "请先刷新并选择服务器模型")
            return

        host = AppConfig.get_server_host()
        web_port = AppConfig.get_server_web_port()
        target_dir = AppConfig.get_model_download_dir()
        filename = model.get("filename", "")

        self.btn_download_model.setEnabled(False)
        self.btn_refresh_models.setEnabled(False)
        self.model_download_progress.setVisible(True)
        self.model_download_progress.setValue(0)
        self.lbl_status.setText(f"下载模型 {filename} 到 {target_dir}...")

        self.model_download_worker = ModelDownloadWorker(host, web_port, filename, target_dir)
        self.model_download_worker.signal_progress.connect(self.model_download_progress.setValue)
        self.model_download_worker.signal_finished.connect(self._on_model_download_finished)
        self.model_download_worker.start()

    def _load_model(self, path):
        """加载 RKNN 模型"""
        self.lbl_status.setText(f"正在加载模型: {os.path.basename(path)}...")
        if self.detector.load_model(path):
            self.lbl_model.setText(f"模型: {os.path.basename(path)} [NPU]")
            self.lbl_model_path.setText(path)
            self.lbl_status.setText(f"模型已加载: {os.path.basename(path)}")
            self.btn_detect.setEnabled(self.current_image is not None)
            AppConfig.set_last_model_path(path)
        else:
            QMessageBox.warning(self, "加载失败",
                f"无法加载模型: {os.path.basename(path)}\n"
                "请确保在 RK3588 平台上运行，且模型文件完整。")
            self.lbl_status.setText("模型加载失败")

    def _on_remote_models_loaded(self, models):
        self.remote_model_combo.clear()
        if not models:
            self.remote_model_combo.addItem("服务器暂无 .rknn 模型")
            self.btn_download_model.setEnabled(False)
            self.lbl_status.setText("服务器暂无 .rknn 模型")
        else:
            for model in models:
                name = model.get("filename", "")
                size_mb = model.get("filesize", 0) / 1024 / 1024
                self.remote_model_combo.addItem(f"{name} ({size_mb:.1f} MB)", model)
            self.btn_download_model.setEnabled(True)
            self.lbl_status.setText(f"发现 {len(models)} 个服务器模型")
        self.btn_refresh_models.setEnabled(True)

    def _on_remote_models_error(self, error_msg):
        self.remote_model_combo.clear()
        self.remote_model_combo.addItem("读取失败")
        self.btn_download_model.setEnabled(False)
        self.btn_refresh_models.setEnabled(True)
        QMessageBox.warning(self, "读取模型失败", error_msg)
        self.lbl_status.setText("读取服务器模型失败")

    def _on_model_download_finished(self, success, message, path):
        self.model_download_progress.setVisible(not success)
        self.btn_refresh_models.setEnabled(True)
        self.btn_download_model.setEnabled(self.remote_model_combo.currentData() is not None)
        if success:
            self.model_download_progress.setValue(100)
            self.lbl_status.setText(f"{message}: {path}")
            self._load_model(path)
        else:
            QMessageBox.warning(self, "下载模型失败", message)
            self.lbl_status.setText(message)

    def start_detection(self):
        """启动检测（异步线程）"""
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先打开一张图片")
            return
        if not self.detector.is_loaded():
            QMessageBox.warning(self, "提示", "请先加载模型")
            return

        self.btn_detect.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.lbl_status.setText("检测中...")
        self.lbl_detection_info.setText("")

        conf = self.conf_spin.value()
        iou = self.iou_spin.value()

        self.detection_worker = DetectionWorker(
            self.detector, self.current_image, conf, iou
        )
        self.detection_worker.signal_done.connect(self._on_detection_done)
        self.detection_worker.signal_error.connect(self._on_detection_error)
        self.detection_worker.start()

    def clear_results(self):
        """清除当前结果"""
        self._close_qwen_session()
        self.current_result = None
        self.current_result_path = None
        self.current_detections = []
        self.btn_upload.setEnabled(False)
        self.btn_qwen_send.setEnabled(False)
        self.btn_qwen_suggest.setEnabled(False)
        self.clear_qwen_chat()
        self.lbl_detection_info.setText("")
        if self.current_image is not None:
            self._display_image(self.current_image)
        self.lbl_status.setText("已清除检测结果")

    def start_upload(self):
        """启动上传（异步线程）"""
        if self.current_result_path is None or not os.path.exists(self.current_result_path):
            QMessageBox.warning(self, "提示", "没有检测结果可上传，请先运行检测")
            return

        host = AppConfig.get_server_host()
        port = AppConfig.get_server_port()

        self.btn_upload.setEnabled(False)
        self.upload_progress.setVisible(True)
        self.upload_progress.setValue(0)
        upload_name = os.path.basename(self.current_result_path)
        self.lbl_upload_status.setText(f"正在上传: {upload_name}")
        self.lbl_status.setText(f"上传 {upload_name} 至 {host}:{port}...")

        self.upload_worker = UploadWorker(
            self.current_result_path, host, port
        )
        self.upload_worker.signal_progress.connect(self.upload_progress.setValue)
        self.upload_worker.signal_finished.connect(self._on_upload_finished)
        self.upload_worker.start()

    def select_qwen_dir(self):
        """选择 Qwen3-VL demo 运行目录"""
        path = QFileDialog.getExistingDirectory(
            self, "选择 Qwen3-VL demo 目录", AppConfig.get_qwen_vl_dir()
        )
        if not path:
            return
        AppConfig.set_qwen_vl_dir(path)
        self.lbl_qwen_dir.setText(path)
        self._close_qwen_session()

    def start_qwen_analysis(self):
        """对当前检测结果图调用本地 Qwen3-VL 分析"""
        if self.current_result_path is None or not os.path.exists(self.current_result_path):
            QMessageBox.warning(self, "提示", "请先完成图片检测")
            return
        question = self.qwen_input.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "提示", "请输入要问 Qwen3-VL 的问题")
            return

        self._pending_qwen_question = question
        self._append_qwen_message("user", question)
        self.qwen_input.clear()
        self.btn_qwen_send.setEnabled(False)
        self.btn_qwen_suggest.setEnabled(False)
        self.btn_qwen_send.setText("分析中...")
        self.lbl_status.setText("Qwen3-VL 分析中...")

        first_turn = self.qwen_session is None or self.qwen_session_image_path != self.current_result_path
        self._ensure_qwen_session()
        send_question = (
            build_qwen_prompt(self.current_detections, [], question)
            if first_turn else question
        )
        self.qwen_worker = QwenAnalysisWorker(self.qwen_session, send_question)
        self.qwen_worker.signal_done.connect(self._on_qwen_done)
        self.qwen_worker.signal_error.connect(self._on_qwen_error)
        self.qwen_worker.start()

    def ask_default_qwen_question(self):
        self.qwen_input.setPlainText("请分析图片中的PCB缺陷类型、可能原因，并给出可执行的生产改进建议。")
        self.start_qwen_analysis()

    def clear_qwen_chat(self):
        self.qwen_history = []
        self._pending_qwen_question = ""
        if hasattr(self, "qwen_output"):
            self.qwen_output.setHtml(
                "<span style='color:#777;'>检测完成后先点“默认分析”查看图片内容，后续可继续追问工艺问题。</span>"
            )

    def _ensure_qwen_session(self):
        if (
            self.qwen_session is not None
            and self.qwen_session_image_path == self.current_result_path
        ):
            return
        self._close_qwen_session()
        self.qwen_session = QwenVLSession(
            AppConfig.get_qwen_vl_dir(),
            self.current_result_path,
            timeout_sec=AppConfig.get_qwen_vl_timeout(),
        )
        self.qwen_session_image_path = self.current_result_path

    def _close_qwen_session(self):
        if self.qwen_session is not None:
            self.qwen_session.close()
        self.qwen_session = None
        self.qwen_session_image_path = ""

    def cleanup(self):
        """清理资源（窗口关闭时调用）"""
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.quit()
            self.detection_worker.wait()
        if self.upload_worker and self.upload_worker.isRunning():
            self.upload_worker.quit()
            self.upload_worker.wait()
        if self.model_list_worker and self.model_list_worker.isRunning():
            self.model_list_worker.quit()
            self.model_list_worker.wait()
        if self.model_download_worker and self.model_download_worker.isRunning():
            self.model_download_worker.quit()
            self.model_download_worker.wait()
        if self.qwen_worker and self.qwen_worker.isRunning():
            self.qwen_worker.quit()
            self.qwen_worker.wait()
        self._close_qwen_session()
        self.detector.unload()

    # ========== 内部方法 ==========

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_result is not None:
            self._display_image(self.current_result)
        elif self.current_image is not None:
            self._display_image(self.current_image)

    def _display_image(self, img_rgb):
        """在 QLabel 中完整显示 RGB 图像，按当前可见区域等比例缩放。"""
        h, w = img_rgb.shape[:2]

        view_w = max(self.image_label.width() - 8, 1)
        view_h = max(self.image_label.height() - 8, 1)
        if view_w <= 16 or view_h <= 16:
            view_w, view_h = 500, 320

        scale = min(view_w / w, view_h / h)
        if scale != 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            display = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)
        else:
            display = img_rgb.copy()

        h2, w2 = display.shape[:2]
        bytes_per_line = 3 * w2
        qimg = QImage(display.data, w2, h2, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def _on_detection_done(self, result):
        """检测完成回调"""
        result_img, detections = result
        self._close_qwen_session()
        self.current_result = result_img
        self.current_detections = detections
        self._display_image(result_img)

        # 更新检测信息
        num = len(detections)
        info_parts = [f"检测到 {num} 个目标"]
        if detections:
            # 统计各类别数量
            classes = {}
            for d in detections:
                name = d.get("class_name", f"class_{d['class_id']}")
                classes[name] = classes.get(name, 0) + 1
            for name, count in classes.items():
                info_parts.append(f"{name}: {count}")

        self.lbl_detection_info.setText(" | ".join(info_parts))

        # 保存为唯一文件名，避免服务器端按 _detect_result.jpg 覆盖旧结果。
        temp_path = build_result_image_path(self.current_image_path)
        cv2.imwrite(temp_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        self.current_result_path = temp_path

        self.btn_detect.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_qwen_send.setEnabled(True)
        self.btn_qwen_suggest.setEnabled(True)
        self.clear_qwen_chat()
        self.lbl_status.setText(f"检测完成 - {info_parts[0]}")

    def _on_detection_error(self, error_msg):
        """检测出错回调"""
        QMessageBox.warning(self, "检测出错", error_msg)
        self.btn_detect.setEnabled(True)
        self.lbl_status.setText("检测出错")

    def _on_upload_finished(self, success, message):
        """上传完成回调"""
        self.upload_progress.setVisible(not success)
        self.lbl_upload_status.setText(message)
        self.btn_upload.setEnabled(True)
        if success:
            self.lbl_status.setText("上传完成")
            self.upload_progress.setValue(100)
        else:
            self.lbl_status.setText("上传失败")

    def _on_qwen_done(self, reply):
        """Qwen3-VL 分析完成"""
        if self._pending_qwen_question:
            self.qwen_history.append(("user", self._pending_qwen_question))
        self.qwen_history.append(("assistant", reply))
        self._pending_qwen_question = ""
        self._append_qwen_message("assistant", reply)
        self.btn_qwen_send.setEnabled(True)
        self.btn_qwen_suggest.setEnabled(True)
        self.btn_qwen_send.setText("发送问题")
        self.lbl_status.setText("Qwen3-VL 分析完成")

    def _on_qwen_error(self, error_msg):
        """Qwen3-VL 分析失败"""
        self._append_qwen_message("system", f"Qwen3-VL 分析失败: {error_msg}")
        self._pending_qwen_question = ""
        self.btn_qwen_send.setEnabled(True)
        self.btn_qwen_suggest.setEnabled(True)
        self.btn_qwen_send.setText("发送问题")
        self.lbl_status.setText("Qwen3-VL 分析失败")

    def _append_qwen_message(self, role, content):
        color = {
            "user": "#007aff",
            "assistant": "#333333",
            "system": "#c0392b",
        }.get(role, "#333333")
        label = {
            "user": "你",
            "assistant": "Qwen",
            "system": "系统",
        }.get(role, role)
        self.qwen_output.append(
            f"<div style='margin:6px 0; color:{color};'>"
            f"<b>{html.escape(label)}:</b> "
            f"<span style='white-space:pre-wrap;'>{html.escape(content)}</span>"
            "</div>"
        )
        scrollbar = self.qwen_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
