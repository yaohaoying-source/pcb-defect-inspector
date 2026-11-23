import argparse
import sys
import os
import cv2
import time
import numpy as np
import threading
import subprocess
import signal

# ==========================================
# ⚙️ 全局配置参数 (PCB 缺陷检测 — RK3588 + OV13855)
# ==========================================
MODEL_PATH = ''
MODEL_INPUT_SIZE = (640, 640)
CONF_THRES = 0.45
IOU_THRES = 0.25

CAP_DEVICE = ''
CAP_WIDTH = 640
CAP_HEIGHT = 480
CAP_FPS = 30

GUI_DISPLAY_SIZE = (720, 720)
BOX_THICKNESS = 2
FONT_SCALE = 0.65
FPS_FONT_SCALE = 1.0
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ——— PCB 缺陷类别 ———
PCB_CLASSES = [
    'Missing_hole',
    'Mouse_bite',
    'Open_circuit',
    'Short',
    'Spur',
    'Spurious_copper'
]

# ——— 每种缺陷独立配色 (BGR 格式) ———
DEFECT_COLORS = {
    0: (0,   0,   255),   # Missing_hole    — 红色
    1: (0,   165, 255),   # Mouse_bite      — 橙色
    2: (0,   255, 255),   # Open_circuit    — 黄色
    3: (255, 0,   0),     # Short           — 蓝色
    4: (255, 0,   255),   # Spur            — 紫色
    5: (0,   255, 0),     # Spurious_copper — 绿色
}
DEFAULT_BOX_COLOR = (50, 50, 255)

TEXT_COLOR = (255, 255, 255)
FPS_COLOR = (0, 255, 80)

# 检测平滑参数
SMOOTH_ALPHA = 0.55
MATCH_IOU_THRES = 0.25
MAX_MISSING_FRAMES = 4
MIN_HITS = 2

# OV13855 抗频闪
POWER_LINE_FREQ = 1


# ==========================================
# 环境变量与线程同步
# ==========================================
DISPLAY = os.environ.get('DISPLAY')
if DISPLAY is None and os.path.exists('/tmp/.X11-unix/X0'):
    DISPLAY = ':0'
    os.environ['DISPLAY'] = DISPLAY
DISPLAY_AVAILABLE = DISPLAY is not None

is_running = True


# ==========================================
# 信号处理（Ctrl+C / kill 优雅退出）
# ==========================================
def _signal_handler(sig, frame):
    global is_running
    print("\n[INFO] 收到退出信号，正在清理...")
    is_running = False

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# 三级流水线同步原语
frame_lock = threading.Lock()
new_frame_event = threading.Event()
_current_frame_rgb = None
shared_boxes = []
shared_fps = 0.0


# ==========================================
# 🚀 GStreamer 管道
# ==========================================
class CameraPipe:

    def __init__(self, device, width, height, fps):
        self.width = width
        self.height = height
        self.bytes_per_frame = width * height * 3
        self.is_running = True

        self._frame_write = np.empty((height, width, 3), dtype=np.uint8)
        self._frame_ready = np.empty((height, width, 3), dtype=np.uint8)
        self._cam_lock = threading.Lock()
        self._has_frame = False

        cmd = [
            "gst-launch-1.0", "-q",
            "v4l2src", f"device={device}",
            f"extra-controls=c,power_line_frequency={POWER_LINE_FREQ}",
            "do-timestamp=true",
            "!",
            f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1",
            "!",
            "queue", "max-size-buffers=3", "leaky=downstream", "!",
            "videoconvert", "n-threads=4", "!",
            "video/x-raw,format=RGB", "!",
            "queue", "max-size-buffers=1", "leaky=downstream", "!",
            "fdsink", "fd=1", "sync=false"
        ]

        self.cmd = cmd
        print(f"[INFO] 启动管道: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.bytes_per_frame * 4
        )

        self.thread = threading.Thread(target=self._read_loop, name="CameraThread")
        self.thread.daemon = True
        self.thread.start()

    def _restart_pipeline(self):
        try:
            self.process.terminate()
            self.process.wait(timeout=2)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        print("[INFO] 正在重启 GStreamer 管道...")
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.bytes_per_frame * 4
        )
        print("[INFO] GStreamer 管道已重连")

    def _read_loop(self):
        frame_size = self.bytes_per_frame
        write_buf = self._frame_write

        while self.is_running:
            pipe = self.process.stdout
            raw = pipe.read(frame_size)

            if len(raw) == 0:
                print("[ERROR] GStreamer 管道断开，3 秒后尝试重连...")
                time.sleep(3.0)
                if not self.is_running:
                    break
                self._restart_pipeline()
                continue

            if len(raw) != frame_size:
                print("[WARN] 帧不完整，重新同步...")
                continue

            np.copyto(
                write_buf.reshape(-1),
                np.frombuffer(raw, dtype=np.uint8)
            )

            with self._cam_lock:
                np.copyto(self._frame_ready, write_buf)
                self._has_frame = True

    def read(self):
        with self._cam_lock:
            if not self._has_frame:
                return False, None
            return True, self._frame_ready.copy()

    def release(self):
        self.is_running = False
        try:
            self.process.terminate()
            self.process.wait(timeout=3)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


# ==========================================
# 简单 IOU 跟踪器
# ==========================================
class SimpleTracker:

    def __init__(self):
        self.tracks = []
        self._next_id = 0

    def update(self, detections):
        if not detections:
            self._age_tracks()
            return self._get_visible_tracks()

        if not self.tracks:
            for d in detections:
                self._create_track(d)
            return self._get_visible_tracks()

        n_t = len(self.tracks)
        n_d = len(detections)
        iou_mat = np.zeros((n_t, n_d), dtype=np.float32)
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                if t['class_id'] != d['class_id']:
                    iou_mat[i, j] = 0.0
                else:
                    iou_mat[i, j] = self._iou(t['sbox'], d['box'])

        matched_t = set()
        matched_d = set()

        pairs = []
        for i in range(n_t):
            for j in range(n_d):
                if iou_mat[i, j] >= MATCH_IOU_THRES:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for _, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)

            t = self.tracks[ti]
            d = detections[di]
            a = SMOOTH_ALPHA
            t['sbox'] = [
                a * d['box'][k] + (1.0 - a) * t['sbox'][k]
                for k in range(4)
            ]
            t['score'] = d['score']
            t['class_id'] = d['class_id']
            t['missing'] = 0
            t['hits'] += 1

        for di in range(n_d):
            if di not in matched_d:
                self._create_track(detections[di])

        for ti in range(n_t):
            if ti not in matched_t:
                self.tracks[ti]['missing'] += 1

        self.tracks = [t for t in self.tracks if t['missing'] < MAX_MISSING_FRAMES]
        return self._get_visible_tracks()

    def _create_track(self, det):
        self.tracks.append({
            'id': self._next_id,
            'sbox': list(det['box']),
            'score': det['score'],
            'class_id': det['class_id'],
            'missing': 0,
            'hits': 1,
        })
        self._next_id += 1

    def _age_tracks(self):
        for t in self.tracks:
            t['missing'] += 1
        self.tracks = [t for t in self.tracks if t['missing'] < MAX_MISSING_FRAMES]

    def _get_visible_tracks(self):
        return [
            {
                'box': [int(round(b)) for b in t['sbox']],
                'score': t['score'],
                'class_id': t['class_id']
            }
            for t in self.tracks
            if t['hits'] >= MIN_HITS and t['missing'] <= 1
        ]

    @staticmethod
    def _iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-6)


# ==========================================
# 核心：YOLOv8 后处理 (全向量化版)
# ==========================================
def process_yolov8_outputs(outputs):
    """
    YOLOv8 输出: [1, (4+num_classes), 8400]
    4 = cx, cy, w, h
    num_classes = 6 (PCB 缺陷)
    """
    try:
        num_classes = len(PCB_CLASSES)

        if not isinstance(outputs, (list, tuple)):
            out = np.asarray(outputs)
            if out.ndim == 3 and out.shape[1] == (4 + num_classes):
                arr = out.reshape(-1, 4 + num_classes) if out.shape[0] == 1 else out.transpose(1, 0)
            else:
                raise ValueError(f'无法推断输出维度: shape={out.shape}')

            cls = arr[:, 4:]
            max_idx = np.argmax(cls, axis=1)
            max_score = cls[np.arange(len(arr)), max_idx]

            mask = max_score > CONF_THRES
            if not np.any(mask):
                return []

            arr_k = arr[mask]
            sc_k = max_score[mask]
            ci_k = max_idx[mask]

            x1 = np.round(arr_k[:, 0] - arr_k[:, 2] / 2.0).astype(np.int32)
            y1 = np.round(arr_k[:, 1] - arr_k[:, 3] / 2.0).astype(np.int32)
            w = np.round(arr_k[:, 2]).astype(np.int32)
            h = np.round(arr_k[:, 3]).astype(np.int32)

            boxes_l = np.stack([x1, y1, w, h], axis=1).tolist()
            scores_l = sc_k.tolist()

            indices = cv2.dnn.NMSBoxes(boxes_l, scores_l, CONF_THRES, IOU_THRES)
            if indices is None or len(indices) == 0:
                return []
            flat = np.array(indices).flatten()
            return [{'box': boxes_l[i], 'score': scores_l[i], 'class_id': int(ci_k[i])} for i in flat]

        else:
            all_boxes = []
            all_scores = []
            all_cls = []
            img_w, img_h = MODEL_INPUT_SIZE

            n = len(outputs)
            i = 0
            while i + 2 < n:
                a_cls = np.asarray(outputs[i + 1])
                a_obj = np.asarray(outputs[i + 2])
                if a_cls.ndim < 3 or a_obj.ndim < 3:
                    i += 3
                    continue

                _, Ccls, H, W = a_cls.shape
                cls_map = a_cls[0]
                obj_map = a_obj[0].reshape((H, W))

                stride_x = img_w / float(W)
                stride_y = img_h / float(H)

                valid_mask = obj_map > 0.01
                vy, vx = np.where(valid_mask)

                if len(vy) > 0:
                    v_obj = obj_map[vy, vx]
                    v_cls = cls_map[:, vy, vx]

                    mi = np.argmax(v_cls, axis=0)
                    ms = v_cls[mi, np.arange(len(vy))]

                    fs = ms * v_obj
                    km = fs >= CONF_THRES

                    if np.any(km):
                        ky = vy[km]; kx = vx[km]
                        ks = fs[km]; kc = mi[km]

                        cx = (kx + 0.5) * stride_x
                        cy = (ky + 0.5) * stride_y
                        bw = stride_x * 6.0
                        bh = stride_y * 6.0

                        x1 = np.round(cx - bw / 2.0).astype(np.int32)
                        y1 = np.round(cy - bh / 2.0).astype(np.int32)
                        bw_i = int(round(bw))
                        bh_i = int(round(bh))

                        nk = len(ks)
                        sb = np.stack([x1, y1,
                                       np.full(nk, bw_i, dtype=np.int32),
                                       np.full(nk, bh_i, dtype=np.int32)], axis=1)
                        all_boxes.append(sb)
                        all_scores.append(ks)
                        all_cls.append(kc)

                i += 3

            if not all_boxes:
                return []

            mb = np.concatenate(all_boxes, axis=0)
            ms = np.concatenate(all_scores, axis=0)
            mc = np.concatenate(all_cls, axis=0)

            bl = mb.tolist()
            sl = ms.tolist()
            indices = cv2.dnn.NMSBoxes(bl, sl, CONF_THRES, IOU_THRES)
            if indices is None or len(indices) == 0:
                return []
            flat = np.array(indices).flatten()
            return [{'box': bl[i], 'score': sl[i], 'class_id': int(mc[i])} for i in flat]

    except Exception as e:
        print(f"[WARN] 后处理异常: {e}")
        return []


# ==========================================
# 标签绘制工具（缺陷类别独立配色）
# ==========================================
def draw_label(img, text, x, y, bg_color, font_scale=FONT_SCALE,
               text_color=TEXT_COLOR, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, font_scale, thickness)
    y_top = max(0, y - th - baseline - 6)
    cv2.rectangle(img, (x, y_top), (x + tw + 8, y), bg_color, -1)
    cv2.putText(img, text, (x + 4, y - baseline - 3),
                LABEL_FONT, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_fps(img, fps_val):
    text = f'NPU FPS: {fps_val:.1f}'
    (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, FPS_FONT_SCALE, 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (20 + tw + 10, 20 + th + baseline), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, text, (20, 20 + th),
                LABEL_FONT, FPS_FONT_SCALE, FPS_COLOR, 2, cv2.LINE_AA)


def draw_defect_summary(img, detections):
    """在画面右上角绘制缺陷统计摘要"""
    if not detections:
        return

    count = {}
    for d in detections:
        cid = d['class_id']
        count[cid] = count.get(cid, 0) + 1

    x_start = img.shape[1] - 260
    y_start = 15
    line_h = 28

    n_lines = len(count) + 1
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x_start - 10, y_start - 5),
                  (img.shape[1] - 10, y_start + line_h * n_lines + 5),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, f"Defects: {len(detections)}",
                (x_start, y_start + 20),
                LABEL_FONT, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    y_cur = y_start + 20 + line_h
    for cid in sorted(count.keys()):
        name = PCB_CLASSES[cid] if 0 <= cid < len(PCB_CLASSES) else f"Class{cid}"
        color = DEFECT_COLORS.get(cid, DEFAULT_BOX_COLOR)
        cv2.putText(img, f"{name}: {count[cid]}",
                    (x_start + 5, y_cur),
                    LABEL_FONT, 0.55, color, 1, cv2.LINE_AA)
        y_cur += line_h


# ==========================================
# 后台 NPU 推理线程
# ==========================================
_infer_buf = [
    np.empty((1, MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), dtype=np.uint8),
    np.empty((1, MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), dtype=np.uint8),
]
_buf_idx = 0


def inference_worker(rknn):
    global shared_fps, is_running, shared_boxes, _current_frame_rgb, _buf_idx

    while is_running:
        if not new_frame_event.wait(timeout=0.5):
            continue
        new_frame_event.clear()

        with frame_lock:
            frame_rgb = _current_frame_rgb

        if frame_rgb is None:
            continue

        cur = _buf_idx
        cv2.resize(frame_rgb, MODEL_INPUT_SIZE, dst=_infer_buf[cur][0],
                   interpolation=cv2.INTER_NEAREST)
        _buf_idx = 1 - cur

        t0 = time.time()
        outputs = rknn.inference(inputs=[_infer_buf[cur]])
        results = process_yolov8_outputs(outputs)
        t1 = time.time()

        shared_fps = 1.0 / (t1 - t0 + 1e-6)
        with frame_lock:
            shared_boxes = results


# ==========================================
# 主函数: PCB 缺陷检测
# ==========================================
def main():
    global MODEL_PATH, CAP_DEVICE
    parser = argparse.ArgumentParser(description="Independent RK3588 GStreamer detection prototype")
    parser.add_argument("--model", required=True, help="Path to RKNN model")
    parser.add_argument("--camera", required=True, help="V4L2 device, e.g. /dev/video0")
    args = parser.parse_args()
    MODEL_PATH, CAP_DEVICE = args.model, args.camera
    from rknnlite.api import RKNNLite
    global is_running, shared_boxes, _current_frame_rgb

    # ---------- 1. 加载模型 ----------
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_PATH) != 0:
        print("[ERROR] 模型加载失败，请检查路径: " + MODEL_PATH)
        return

    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print(f"[WARN] NPU 三核模式失败(ret={ret})，回退到自动模式...")
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            print("[ERROR] NPU 初始化失败")
            return
    else:
        print("[INFO] ✅ RK3588 NPU 三核全开")

    print(f"[INFO] 模型: {MODEL_PATH}")
    print(f"[INFO] 检测类别 ({len(PCB_CLASSES)}): {PCB_CLASSES}")

    # ---------- 2. 启动推理线程 ----------
    infer_thread = threading.Thread(target=inference_worker, args=(rknn,),
                                    name="NPU-InferThread")
    infer_thread.daemon = True
    infer_thread.start()

    # ---------- 3. 启动摄像头管道 ----------
    cap = CameraPipe(CAP_DEVICE, CAP_WIDTH, CAP_HEIGHT, CAP_FPS)

    # ---------- 4. 初始化跟踪器 ----------
    tracker = SimpleTracker()

    window_name = 'PCB Defect Detection - YOLOv8 + RK3588'
    window_created = False
    scale_x = GUI_DISPLAY_SIZE[0] / MODEL_INPUT_SIZE[0]
    scale_y = GUI_DISPLAY_SIZE[1] / MODEL_INPUT_SIZE[1]

    print("[INFO] 开始 PCB 缺陷实时检测... (按 ESC 键退出)")
    print(f"[INFO] 检测平滑: alpha={SMOOTH_ALPHA}, min_hits={MIN_HITS}, max_missing={MAX_MISSING_FRAMES}")

    time.sleep(1.0)

    while is_running:
        ret, frame_rgb = cap.read()

        if not ret or frame_rgb is None:
            if window_created and DISPLAY_AVAILABLE:
                cv2.waitKey(10)
            else:
                time.sleep(0.01)
            continue

        with frame_lock:
            _current_frame_rgb = frame_rgb
            raw_boxes = shared_boxes

        new_frame_event.set()

        smoothed_boxes = tracker.update(raw_boxes)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_display = cv2.resize(frame_bgr, GUI_DISPLAY_SIZE,
                                   interpolation=cv2.INTER_NEAREST)

        # 绘制检测框（每种缺陷独立颜色）
        for item in smoothed_boxes:
            x_npu, y_npu, w_npu, h_npu = item['box']
            score = item['score']
            class_id = item['class_id']

            x = int(x_npu * scale_x)
            y = int(y_npu * scale_y)
            w = int(w_npu * scale_x)
            h = int(h_npu * scale_y)

            class_name = (PCB_CLASSES[class_id]
                          if 0 <= class_id < len(PCB_CLASSES)
                          else f"Defect_{class_id}")

            box_color = DEFECT_COLORS.get(class_id, DEFAULT_BOX_COLOR)

            cv2.rectangle(frame_display, (x, y), (x + w, y + h),
                          box_color, BOX_THICKNESS)
            draw_label(frame_display, f"{class_name} {score:.0%}", x, y,
                       bg_color=box_color)

        draw_fps(frame_display, shared_fps)
        draw_defect_summary(frame_display, smoothed_boxes)

        if DISPLAY_AVAILABLE:
            if not window_created:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name,
                                     GUI_DISPLAY_SIZE[0], GUI_DISPLAY_SIZE[1])
                except Exception:
                    pass
                window_created = True

            cv2.imshow(window_name, frame_display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.01)

    # ---------- 安全退出 ----------
    print("\n[INFO] 正在安全退出...")
    is_running = False
    new_frame_event.set()
    try:
        infer_thread.join(timeout=2.0)
    except Exception:
        pass
    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        rknn.release()
    except Exception:
        pass
    print("[INFO] PCB 缺陷检测退出完成！")


if __name__ == '__main__':
    main()
