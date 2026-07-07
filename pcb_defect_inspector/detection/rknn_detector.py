"""
RKNN (.rknn) 检测器 — 来自 yolo_app 升级版
双缓冲推理 + 三核 NPU + 全向量化后处理
"""
import os
import platform
import numpy as np
import cv2

from .base_detector import BaseDetector
from .postprocess import process_yolo_output

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

PCB_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short_circuit",
    "spur",
    "spurious_copper",
]

_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 0),
    (255, 0, 255), (0, 255, 0), (128, 0, 128), (255, 165, 0),
    (0, 128, 128), (128, 128, 0), (72, 61, 139), (0, 191, 255),
    (34, 139, 34), (255, 20, 147), (218, 165, 32), (128, 0, 0),
]


def _can_import_rknn() -> bool:
    """检测能否导入 rknnlite（比读 cpuinfo 更可靠）"""
    try:
        import rknnlite  # noqa: F401
        return True
    except ImportError:
        return False


class RknnDetector(BaseDetector):
    """基于 RKNN NPU 的检测器 — 双缓冲 + 三核全开"""

    def __init__(self):
        self.rknn = None
        self.model_path = ""
        self.input_size = (640, 640)
        self.num_classes = 80
        self._num_classes_known = False

        # 双缓冲推理输入
        self._infer_buf = [
            np.empty((1, self.input_size[1], self.input_size[0], 3), dtype=np.uint8),
            np.empty((1, self.input_size[1], self.input_size[0], 3), dtype=np.uint8),
        ]
        self._buf_idx = 0

    def load_model(self, model_path: str) -> bool:
        if not _can_import_rknn():
            print("[RknnDetector] 未安装 rknnlite，无法加载 RKNN 模型")
            return False

        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            if self.rknn.load_rknn(model_path) != 0:
                print("[RknnDetector] 模型加载失败")
                self.rknn = None
                return False

            # 三核全开（来自 yolo_app），失败回退到自动
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                print(f"[RknnDetector] 三核模式失败(ret={ret})，回退自动模式...")
                if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                    print("[RknnDetector] NPU 初始化失败")
                    self.rknn = None
                    return False
            else:
                print("[RknnDetector] RK3588 NPU 三核全开")

            self.model_path = model_path
            return True

        except ImportError:
            print("[RknnDetector] 请安装 rknnlite")
            return False
        except Exception as e:
            print(f"[RknnDetector] 加载异常: {e}")
            self.rknn = None
            return False

    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45,
               draw: bool = True):
        """执行一次 RKNN 检测。

        Args:
            image: RGB uint8 图像，shape 为 [H, W, 3]。
            conf: 置信度阈值。
            iou: NMS IOU 阈值。
            draw: 是否返回已画框图像。实时摄像头建议 False，跟踪后再画框。

        Returns:
            draw=True:  (result_img, detections)
            draw=False: (image, detections)，不做 image.copy()/画框，降低每帧开销。
        """
        if self.rknn is None:
            raise RuntimeError("RKNN 模型未加载")

        h, w = image.shape[:2]

        # 双缓冲：写入当前缓冲区，下次用另一个。
        cur = self._buf_idx
        cv2.resize(image, self.input_size, dst=self._infer_buf[cur][0],
                   interpolation=cv2.INTER_NEAREST)
        self._buf_idx = 1 - cur

        outputs = self.rknn.inference(inputs=[self._infer_buf[cur]])

        # 输出结构固定，类别数只需推断一次。
        if not self._num_classes_known:
            self.num_classes = self._infer_num_classes(outputs)
            self._num_classes_known = True

        raw_detections = process_yolo_output(
            outputs, conf_thres=conf, iou_thres=iou,
            input_size=self.input_size, num_classes=self.num_classes
        )

        scale_x = w / self.input_size[0]
        scale_y = h / self.input_size[1]
        class_names = PCB_CLASSES if self.num_classes == len(PCB_CLASSES) else COCO_CLASSES

        detections = []
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)

            cls_id = int(det["class_id"])
            score = float(det["confidence"])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": score,
                "class_id": cls_id,
                "class_name": cls_name
            })

        if not draw:
            return image, detections

        return self.draw_detections(image, detections, copy_image=True), detections

    def draw_detections(self, image: np.ndarray, detections, copy_image: bool = True):
        """按检测结果画框。单独拆出，便于 tracker 平滑后再画框。"""
        result_img = image.copy() if copy_image else image

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = int(det["class_id"])
            score = float(det.get("confidence", 0.0))
            cls_name = det.get("class_name", f"class_{cls_id}")

            color = _COLORS[cls_id % len(_COLORS)]
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y0 = max(0, y1 - th - 6)
            cv2.rectangle(result_img, (x1, y0), (x1 + tw + 6, y1), color, -1)
            cv2.putText(result_img, label, (x1 + 3, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return result_img

    def is_loaded(self) -> bool:
        return self.rknn is not None

    @staticmethod
    def _infer_num_classes(outputs) -> int:
        """Infer class count from common RKNN YOLO output layouts."""
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 1:
                out = np.asarray(outputs[0])
                if out.ndim == 3:
                    return max(int(out.shape[1]) - 4, 1)
                if out.ndim == 4:
                    return max(int(out.shape[1]) - 4, 1)
            if len(outputs) >= 2:
                cls = np.asarray(outputs[1])
                if cls.ndim == 4:
                    return int(cls.shape[1])
        return 80

    def is_platform_supported(self) -> bool:
        return _can_import_rknn()

    def unload(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
