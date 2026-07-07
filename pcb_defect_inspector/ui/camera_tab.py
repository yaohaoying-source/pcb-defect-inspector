"""
摄像头实时检测标签页 - 视频流推理 + 定时截图 + 选择性上传
"""
import os
import cv2
import time
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QScrollArea, QMessageBox,
    QProgressBar, QListWidget, QListWidgetItem, QAbstractItemView,
    QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage

from utils.config import AppConfig
from detection.rknn_detector import RknnDetector
from detection.tracker import SimpleTracker


class GstCameraCapture:
    """GStreamer appsink capture for RKISP V4L2 nodes.

    OpenCV on RKISP may probe unsupported BGR/RGB/YUYV formats before the
    application can set UYVY, which can block stream startup. This class opens
    the exact V4L2 raw format and asks appsink to keep only the newest frame.
    """
    outputs_rgb = True

    def __init__(self, device, width=640, height=480, fps=30):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._gst = None
        self._pipeline = None
        self._sink = None
        self._opened = False

    def open(self):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception:
            return False

        Gst.init(None)
        self._gst = Gst
        pipeline_desc = (
            f"v4l2src device={self.device} io-mode=mmap ! "
            f"video/x-raw,format=NV12,width={self.width},height={self.height},"
            f"framerate={self.fps}/1 ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            "videoconvert ! video/x-raw,format=RGB ! "
            "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
        )
        try:
            self._pipeline = Gst.parse_launch(pipeline_desc)
            self._sink = self._pipeline.get_by_name("sink")
            if self._sink is None:
                self.release()
                return False
            self._pipeline.set_state(Gst.State.PLAYING)
            ret, _, _ = self._pipeline.get_state(2 * Gst.SECOND)
            if ret == Gst.StateChangeReturn.FAILURE:
                self.release()
                return False
            ok, frame = self.read(timeout_ms=1500)
            if ok and frame is not None:
                self._opened = True
                return True
        except Exception:
            pass

        self.release()
        return False

    def isOpened(self):
        return self._opened

    def read(self, timeout_ms=1000):
        if self._sink is None or self._gst is None:
            return False, None
        sample = self._sink.emit("try-pull-sample", int(timeout_ms * self._gst.MSECOND))
        if sample is None:
            return False, None

        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")
        buffer = sample.get_buffer()
        ok, map_info = buffer.map(self._gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            data = np.frombuffer(map_info.data, dtype=np.uint8)
            expected = int(width) * int(height) * 3
            if data.size < expected:
                return False, None
            frame = data[:expected].reshape((int(height), int(width), 3)).copy()
            return True, frame
        finally:
            buffer.unmap(map_info)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def release(self):
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._sink = None
        self._pipeline = None
        self._opened = False


class UploadWorker(QThread):
    """异步上传工作器"""
    signal_progress = pyqtSignal(int)
    signal_finished = pyqtSignal(bool, str)

    def __init__(self, image_paths, host, port):
        """image_paths: 单个路径(str)或多个路径(list)"""
        super().__init__()
        self.image_paths = [image_paths] if isinstance(image_paths, str) else image_paths
        self.host = host
        self.port = port

    def run(self):
        from upload.uploader import upload_file
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            try:
                def make_cb(idx, base):
                    def cb(v):
                        self.signal_progress.emit(base + v // total)
                    return cb
                upload_file(path, self.host, self.port, progress_callback=make_cb(i, i * 100 // total))
                if i == total - 1:
                    self.signal_finished.emit(True, f"上传完成 ({total}个文件)")
            except Exception as e:
                self.signal_finished.emit(False, f"第{i+1}个上传失败: {e}")
                return


class CameraWorker(QThread):
    """摄像头采集 + 推理线程优化版。

    优化点：
    1. detect(draw=False)，避免推理阶段每帧 copy/画框。
    2. tracker 平滑后再画框，显示结果与检测信息一致。
    3. UI signal 默认不限制，便于观察真实实时显示能力。
    4. 使用 time.monotonic() 统计 FPS。
    5. 摄像头可直接使用 /dev/videoXX 路径，避免 index 打开到错误节点。
    """
    signal_frame = pyqtSignal(object, object, float, int, int)
    signal_error = pyqtSignal(str)
    signal_status = pyqtSignal(str)

    def __init__(self, detector, camera_id=0, conf=0.25, iou=0.45,
                 max_width=640, max_height=480, use_tracker=True,
                 display_interval_ms=66, infer_stride=1):
        super().__init__()
        self.detector = detector
        self.camera_id = camera_id
        self.conf = conf
        self.iou = iou
        self.max_width = max_width
        self.max_height = max_height
        self.use_tracker = use_tracker
        self.display_interval_s = max(0.0, display_interval_ms / 1000.0)
        self.infer_stride = max(1, int(infer_stride))
        self._running = False

    def _open_camera(self):
        # camera_id 可以是 int，也可以是 '/dev/video11' 这样的路径。
        # RKISP 节点优先走 GStreamer，避免 OpenCV 在 open 阶段探测不支持的格式。
        if isinstance(self.camera_id, str) and self.camera_id.startswith("/dev/video"):
            gst_cap = GstCameraCapture(
                self.camera_id, width=self.max_width, height=self.max_height, fps=30)
            if gst_cap.open():
                return gst_cap

        # 回退路径：不设置 MJPG，显式用 UYVY，减少格式协商异常。
        for api in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture()
            try:
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
            except Exception:
                pass
            cap.open(self.camera_id, api)
            if not cap.isOpened():
                cap.release()
                continue

            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"UYVY"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # 预读一帧，过滤会打开但读不到数据的节点。
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
        return None

    def run(self):
        self._running = True
        tracker = SimpleTracker() if self.use_tracker else None

        cap = self._open_camera()
        if cap is None or not cap.isOpened():
            self.signal_error.emit(f"无法打开摄像头 (id={self.camera_id})")
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.signal_status.emit(
            f"摄像头: {self.camera_id} | {actual_w}x{actual_h} | "
            f"{'[T] 跟踪器开' if self.use_tracker else '跟踪器关'} | "
            f"UI刷新:{'不限' if self.display_interval_s == 0 else str(int(1/self.display_interval_s)) + 'fps'}"
        )

        infer_count = 0
        loop_count = 0
        t_fps = time.monotonic()
        last_emit_t = 0.0
        fps_smooth = 0.0
        last_error_t = 0.0
        consecutive_read_failures = 0
        last_raw_detections = []
        last_smoothed = []

        while self._running:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                consecutive_read_failures += 1
                now = time.monotonic()
                if now - last_error_t > 2.0:
                    self.signal_status.emit(
                        f"摄像头读帧失败: {self.camera_id} ({consecutive_read_failures})"
                    )
                    last_error_t = now
                if consecutive_read_failures >= 30:
                    self.signal_error.emit("摄像头连续读帧失败，已停止。请重新启动摄像头。")
                    break
                time.sleep(0.01)
                continue

            consecutive_read_failures = 0
            loop_count += 1
            if getattr(cap, "outputs_rgb", False):
                frame_rgb = frame_bgr
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            do_infer = (loop_count % self.infer_stride == 0)
            if do_infer:
                try:
                    # draw=False：只做推理和后处理，不在 detector 内部画框。
                    _, raw_detections = self.detector.detect(
                        frame_rgb, conf=self.conf, iou=self.iou, draw=False)
                    last_raw_detections = raw_detections
                except Exception as e:
                    now = time.monotonic()
                    if now - last_error_t > 2.0:
                        self.signal_status.emit(f"推理异常: {e}")
                        last_error_t = now
                    raw_detections = []
                    last_raw_detections = []

                if tracker:
                    last_smoothed = tracker.update(last_raw_detections)
                else:
                    last_smoothed = last_raw_detections
                infer_count += 1
            else:
                # 跳帧模式下沿用上一帧结果。默认 infer_stride=1，不会走到这里。
                raw_detections = last_raw_detections

            now = time.monotonic()
            elapsed = now - t_fps
            if elapsed >= 1.0:
                instant_fps = infer_count / elapsed
                # Report the current one-second inference rate directly.
                # UI/display FPS is calculated separately in the main Qt thread.
                fps_smooth = instant_fps
                infer_count = 0
                t_fps = now

            if self.display_interval_s == 0 or now - last_emit_t >= self.display_interval_s:
                # tracker 后再画框，减少 detector 内部 copy/画框，并让画面与 smoothed 结果一致。
                if hasattr(self.detector, "draw_detections"):
                    result_img = self.detector.draw_detections(frame_rgb, last_smoothed, copy_image=True)
                else:
                    result_img = frame_rgb.copy()

                self.signal_frame.emit(result_img, last_smoothed, fps_smooth,
                                       len(last_raw_detections), len(last_smoothed))
                last_emit_t = now

        cap.release()
        self.signal_status.emit("摄像头已停止")

    def stop(self):
        self._running = False


class CameraTab(QWidget):
    """实时摄像头检测标签页"""

    SCREENSHOT_DIR = "screenshots"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.detector = RknnDetector()
        self.camera_worker = None
        self.upload_worker = None
        self._streaming = False
        self._current_result_img = None
        self._use_tracker = True  # 默认开启跟踪器
        self._last_frame_t = 0.0
        self._realtime_fps = 0.0

        # 截图记录
        self._screenshot_records = []
        # 自动截图
        self._auto_mode = False
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._auto_screenshot_tick)
        self._auto_elapsed = 0
        self._auto_interval = 5   # 默认5秒
        self._auto_duration = 60  # 默认60秒

        os.makedirs(self.SCREENSHOT_DIR, exist_ok=True)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ---- 工具栏 ----
        toolbar = QHBoxLayout()
        self.btn_start = QPushButton("开始检测")
        self.btn_start.clicked.connect(self.toggle_stream)
        self.btn_start.setMinimumWidth(100)
        toolbar.addWidget(self.btn_start)

        self.btn_screenshot = QPushButton("截图")
        self.btn_screenshot.clicked.connect(self._screenshot)
        self.btn_screenshot.setEnabled(False)
        toolbar.addWidget(self.btn_screenshot)

        self.btn_upload = QPushButton("上传当前帧")
        self.btn_upload.clicked.connect(lambda: self._upload(current=True))
        self.btn_upload.setEnabled(False)
        toolbar.addWidget(self.btn_upload)

        toolbar.addWidget(QLabel("摄像头:"))
        self.cam_combo = QComboBox()
        # 扫描 /dev/video* 设备。itemData 保存设备路径，避免 OpenCV 按 index 打开到错误节点。
        import glob as _glob
        import re as _re
        devices = sorted(
            (d for d in _glob.glob("/dev/video*") if _re.match(r"/dev/video\d+$", d)),
            key=lambda x: int(x.replace("/dev/video", ""))
        )
        if devices:
            for dev in devices:
                self.cam_combo.addItem(dev, dev)
            # 根据你当前板端扫描结果，/dev/video11 是可读节点；存在时优先选中。
            preferred = "/dev/video11"
            if preferred in devices:
                self.cam_combo.setCurrentIndex(devices.index(preferred))
        else:
            for i in range(12):
                self.cam_combo.addItem(f"摄像头 {i}", i)
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        toolbar.addWidget(self.cam_combo)
        toolbar.addStretch()

        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("font-weight: bold; font-size: 16px; color: #27ae60;")
        toolbar.addWidget(self.lbl_fps)
        toolbar.addStretch()

        self.lbl_model = QLabel("模型: 未加载")
        self.lbl_model.setStyleSheet("color: #888;")
        toolbar.addWidget(self.lbl_model)
        layout.addLayout(toolbar)

        # ---- 视频预览区域 ----
        self.video_label = QLabel("加载模型后点击「开始检测」启动实时推理")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(500, 300)
        self.video_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border: 2px solid #333; color: #888; }")
        scroll = QScrollArea()
        scroll.setWidget(self.video_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        # ---- 参数面板（可折叠）----
        self._create_param_panel()
        layout.insertWidget(2, self.param_panel)

        # ---- 截图记录面板（可折叠）----
        self._create_screenshot_panel()
        layout.addWidget(self.screenshot_panel)

        # ---- 底部状态栏 ----
        status_layout = QHBoxLayout()
        self.lbl_status = QLabel("就绪 - 请先加载模型，然后启动摄像头检测")
        self.lbl_status.setStyleSheet("color: #666; padding: 4px;")
        status_layout.addWidget(self.lbl_status, stretch=1)
        self.lbl_detection_info = QLabel("")
        status_layout.addWidget(self.lbl_detection_info)
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.clicked.connect(self._load_model_dialog)
        status_layout.addWidget(self.btn_load_model)
        self.btn_params = QPushButton("参数")
        self.btn_params.clicked.connect(lambda: self.param_panel.setVisible(not self.param_panel.isVisible()))
        status_layout.addWidget(self.btn_params)
        self.btn_records = QPushButton("截图记录")
        self.btn_records.clicked.connect(lambda: self.screenshot_panel.setVisible(not self.screenshot_panel.isVisible()))
        status_layout.addWidget(self.btn_records)
        layout.addLayout(status_layout)

        # ---- 上传进度条 ----
        up_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(16)
        self.lbl_upload_status = QLabel("")
        up_layout.addWidget(self.progress_bar)
        up_layout.addWidget(self.lbl_upload_status)
        layout.addLayout(up_layout)

    def _create_param_panel(self):
        """检测参数 + 自动截图设置"""
        panel = QGroupBox("检测参数 & 自动截图")
        form = QFormLayout(panel)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(AppConfig.get_confidence())
        form.addRow("置信度:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(AppConfig.get_iou())
        form.addRow("IOU:", self.iou_spin)

        # 跟踪器开关
        from PyQt5.QtWidgets import QCheckBox as _QCB
        self.tracker_cb = _QCB("启用跟踪器 (IOU匹配+EMA平滑)")
        self.tracker_cb.setChecked(self._use_tracker)
        self.tracker_cb.stateChanged.connect(self._on_tracker_changed)
        form.addRow("跟踪器:", self.tracker_cb)

        # 自动截图设置
        auto_layout = QHBoxLayout()
        self.auto_interval_spin = QSpinBox()
        self.auto_interval_spin.setRange(1, 3600)
        self.auto_interval_spin.setValue(self._auto_interval)
        self.auto_interval_spin.setSuffix(" 秒/张")
        auto_layout.addWidget(QLabel("间隔:"))
        auto_layout.addWidget(self.auto_interval_spin)

        self.auto_duration_spin = QSpinBox()
        self.auto_duration_spin.setRange(5, 86400)
        self.auto_duration_spin.setValue(self._auto_duration)
        self.auto_duration_spin.setSuffix(" 秒")
        auto_layout.addWidget(QLabel("上限:"))
        auto_layout.addWidget(self.auto_duration_spin)

        self.btn_auto = QPushButton("开始自动截图")
        self.btn_auto.clicked.connect(self._toggle_auto)
        self.btn_auto.setEnabled(False)
        auto_layout.addWidget(self.btn_auto)
        form.addRow("自动截图:", auto_layout)

        # 倒计时
        self.lbl_auto_countdown = QLabel("")
        self.lbl_auto_countdown.setStyleSheet("color: #e67e22; font-weight: bold;")
        form.addRow("", self.lbl_auto_countdown)

        self.param_panel = panel
        self.param_panel.hide()

    def _create_screenshot_panel(self):
        """截图记录列表"""
        panel = QGroupBox("截图记录（勾选后上传）")
        layout = QVBoxLayout(panel)

        self.screenshot_list = QListWidget()
        self.screenshot_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.screenshot_list.setMaximumHeight(100)
        layout.addWidget(self.screenshot_list)

        btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(lambda: self._toggle_all(True))
        btn_layout.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton("取消全选")
        self.btn_deselect_all.clicked.connect(lambda: self._toggle_all(False))
        btn_layout.addWidget(self.btn_deselect_all)

        self.btn_upload_selected = QPushButton("⬆ 上传选中")
        self.btn_upload_selected.clicked.connect(lambda: self._upload(current=False))
        self.btn_upload_selected.setEnabled(False)
        btn_layout.addWidget(self.btn_upload_selected)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.screenshot_panel = panel
        self.screenshot_panel.hide()

    # ==================== 流控 ====================

    def toggle_stream(self):
        if self._streaming:
            self._stop_auto()
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self):
        if not self.detector.is_loaded():
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        self.camera_worker = CameraWorker(
            self.detector,
            camera_id=self.cam_combo.currentData(),
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            use_tracker=self._use_tracker,
            display_interval_ms=0,    # UI 不限频，直接体现实时处理能力
            infer_stride=1            # 可改 2：每 2 帧推理一次，进一步降低压力
        )
        self.camera_worker.signal_frame.connect(self._on_frame)
        self.camera_worker.signal_error.connect(self._on_error)
        self.camera_worker.signal_status.connect(self._on_status)
        self._last_frame_t = 0.0
        self._realtime_fps = 0.0
        self.camera_worker.start()
        self._streaming = True
        self.btn_start.setText("停止检测")
        self.btn_start.setStyleSheet("QPushButton { color: #e74c3c; font-weight: bold; }")
        self.btn_screenshot.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_auto.setEnabled(True)
        self.cam_combo.setEnabled(False)
        self.btn_load_model.setEnabled(False)

    def _stop_stream(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker.wait(3000)
        self._streaming = False
        self.btn_start.setText("开始检测")
        self.btn_start.setStyleSheet("")
        self.btn_screenshot.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.btn_auto.setEnabled(False)
        self.cam_combo.setEnabled(True)
        self.btn_load_model.setEnabled(True)
        self.lbl_fps.setText("FPS: --")

    # ==================== 帧处理 ====================

    def _on_frame(self, result_img, detections, _worker_fps, raw_count=0, smoothed_count=0):
        # 实时显示 FPS：按 UI 实际收到并绘制的连续帧间隔计算。
        now = time.monotonic()
        if self._last_frame_t > 0:
            dt = now - self._last_frame_t
            if dt > 0:
                instant_fps = 1.0 / dt
                self._realtime_fps = (
                    instant_fps if self._realtime_fps <= 0
                    else 0.8 * self._realtime_fps + 0.2 * instant_fps
                )
        self._last_frame_t = now

        # 不再每帧 copy；截图保存时再做颜色转换/写盘。
        self._current_result_img = result_img

        h, w = result_img.shape[:2]
        max_w = self.video_label.width()
        max_h = self.video_label.height()
        if max_w > 50 and max_h > 50:
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                nw, nh = int(w * scale), int(h * scale)
                display = cv2.resize(result_img, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                display = result_img
        else:
            display = result_img

        if not display.flags["C_CONTIGUOUS"]:
            display = np.ascontiguousarray(display)

        h2, w2 = display.shape[:2]
        qimg = QImage(display.data, w2, h2, 3 * w2, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        tracker_tag = "[T]" if self._use_tracker else ""
        self.lbl_fps.setText(f"FPS: {self._realtime_fps:.1f} {tracker_tag}")
        if detections:
            classes = {}
            for d in detections:
                name = d.get("class_name", f"cls_{d['class_id']}")
                classes[name] = classes.get(name, 0) + 1
            parts = [f"{name}:{cnt}" for name, cnt in sorted(classes.items())]
            info = " | ".join(parts[:8])
            if self._use_tracker and raw_count > 0:
                info += f"  (原始:{raw_count} 跟踪:{smoothed_count})"
            self.lbl_detection_info.setText(info)
        else:
            self.lbl_detection_info.setText("")

    def _on_error(self, msg):
        QMessageBox.warning(self, "摄像头错误", msg)
        self._stop_auto()
        self._stop_stream()

    def _on_status(self, msg):
        self.lbl_status.setText(msg)

    def _on_camera_changed(self):
        if self._streaming:
            self._stop_stream()
            self._start_stream()

    def _on_tracker_changed(self, state):
        self._use_tracker = (state == Qt.Checked)
        if self._streaming:
            # 重启流以应用更改
            self._stop_stream()
            self._start_stream()

    # ==================== 模型加载 ====================

    def _load_model_dialog(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", AppConfig.get_last_model_path(),
            "RKNN 模型 (*.rknn);;所有文件 (*.*)")
        if not path:
            return
        if self.detector.load_model(path):
            self.lbl_model.setText(f"模型: {os.path.basename(path)} [NPU]")
            self.lbl_status.setText("模型已加载 - 点击「开始检测」启动实时推理")
            AppConfig.set_last_model_path(path)
        else:
            QMessageBox.warning(self, "加载失败",
                f"无法加载模型: {os.path.basename(path)}\n"
                "请确保在 RK3588 平台上运行，且模型文件完整。")
            self.lbl_status.setText("模型加载失败")

    # ==================== 截图 ====================

    def _screenshot(self):
        """手动截图"""
        path = self._save_current_frame()
        if path:
            self._add_record(path)
            self.lbl_status.setText(f"截图已保存: {os.path.basename(path)}")
            QMessageBox.information(self, "截图", f"已保存至:\n{os.path.abspath(path)}")

    def _save_current_frame(self):
        """保存当前帧到文件，返回路径"""
        if self._current_result_img is None:
            QMessageBox.warning(self, "提示", "暂无画面，请先启动检测")
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.SCREENSHOT_DIR, f"coco_detect_{timestamp}.png")
        img_bgr = cv2.cvtColor(self._current_result_img, cv2.COLOR_RGB2BGR)
        if cv2.imwrite(filename, img_bgr):
            return filename
        return None

    def _add_record(self, filepath):
        """添加截图记录到列表"""
        ts = time.strftime("%H:%M:%S")
        self._screenshot_records.append(filepath)
        item = QListWidgetItem()
        cb = QCheckBox(os.path.basename(filepath))
        cb.setChecked(True)  # 默认勾选
        cb.setStyleSheet("font-size: 12px;")
        cb.stateChanged.connect(self._on_check_changed)
        self.screenshot_list.insertItem(0, item)
        self.screenshot_list.setItemWidget(item, cb)
        self._on_check_changed()

    def _on_check_changed(self):
        """任一勾选变化时更新上传按钮状态"""
        self.btn_upload_selected.setEnabled(self._get_checked_paths() != [])

    def _get_checked_paths(self):
        """获取勾选的截图路径列表"""
        paths = []
        for i in range(self.screenshot_list.count()):
            item = self.screenshot_list.item(i)
            cb = self.screenshot_list.itemWidget(item)
            if isinstance(cb, QCheckBox) and cb.isChecked():
                for p in self._screenshot_records:
                    if os.path.basename(p) == cb.text():
                        paths.append(p)
                        break
        return paths

    def _toggle_all(self, checked):
        for i in range(self.screenshot_list.count()):
            cb = self.screenshot_list.itemWidget(self.screenshot_list.item(i))
            if isinstance(cb, QCheckBox):
                cb.setChecked(checked)
        self._on_check_changed()

    # ==================== 自动截图 ====================

    def _toggle_auto(self):
        if self._auto_mode:
            self._stop_auto()
        else:
            self._start_auto()

    def _start_auto(self):
        self._auto_interval = self.auto_interval_spin.value()
        self._auto_duration = self.auto_duration_spin.value()
        self._auto_elapsed = 0
        self._auto_mode = True
        self._auto_timer.start(self._auto_interval * 1000)
        self.btn_auto.setText("停止自动截图")
        self.btn_auto.setStyleSheet("QPushButton { color: #e74c3c; font-weight: bold; }")
        self.auto_interval_spin.setEnabled(False)
        self.auto_duration_spin.setEnabled(False)
        self._update_countdown()
        # 立即拍第一张
        self._auto_screenshot_tick()

    def _stop_auto(self):
        self._auto_mode = False
        self._auto_timer.stop()
        self.btn_auto.setText("开始自动截图")
        self.btn_auto.setStyleSheet("")
        self.auto_interval_spin.setEnabled(True)
        self.auto_duration_spin.setEnabled(True)
        self.lbl_auto_countdown.setText("")

    def _auto_screenshot_tick(self):
        """定时器触发：截图并检查是否到时间上限"""
        if not self._auto_mode or not self._streaming:
            return
        path = self._save_current_frame()
        if path:
            self._add_record(path)
        self._auto_elapsed += self._auto_interval
        if self._auto_elapsed >= self._auto_duration:
            self.lbl_status.setText(f"自动截图完成：共 {self._auto_elapsed} 秒")
            QMessageBox.information(self, "自动截图",
                                    f"已完成！\n时长: {self._auto_elapsed} 秒\n"
                                    f"文件保存在: {os.path.abspath(self.SCREENSHOT_DIR)}")
            self._stop_auto()
        else:
            self._update_countdown()

    def _update_countdown(self):
        remaining = self._auto_duration - self._auto_elapsed
        self.lbl_auto_countdown.setText(
            f"自动截图中... 剩余 {remaining} 秒 / 共 {self._auto_duration} 秒")

    # ==================== 上传 ====================

    def _upload(self, current=True):
        """
        current=True:  上传当前帧
        current=False: 上传截图记录中勾选的文件
        """
        host = AppConfig.get_server_host()
        port = AppConfig.get_server_port()

        if current:
            # 先保存当前帧
            path = self._save_current_frame()
            if not path:
                return
            paths = [path]
            self.lbl_upload_status.setText("正在上传当前帧...")
        else:
            paths = self._get_checked_paths()
            if not paths:
                QMessageBox.warning(self, "提示", "请先在截图记录中勾选要上传的文件")
                return
            self.lbl_upload_status.setText(f"正在上传 {len(paths)} 个文件...")

        self.btn_upload.setEnabled(False)
        self.btn_upload_selected.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"上传至 {host}:{port}...")

        self.upload_worker = UploadWorker(paths, host, port)
        self.upload_worker.signal_progress.connect(self.progress_bar.setValue)
        self.upload_worker.signal_finished.connect(self._on_upload_finished)
        self.upload_worker.start()

    def _on_upload_finished(self, success, message):
        self.progress_bar.setVisible(not success)
        self.lbl_upload_status.setText(message)
        self.btn_upload.setEnabled(True)
        self._on_check_changed()
        if success:
            self.lbl_status.setText(message)
            self.progress_bar.setValue(100)
        else:
            self.lbl_status.setText("上传失败")
            QMessageBox.warning(self, "上传失败", message)

    # ==================== 清理 ====================

    def cleanup(self):
        self._stop_auto()
        self._stop_stream()
        self.detector.unload()
