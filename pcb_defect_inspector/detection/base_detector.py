"""
检测器抽象基类 - 定义统一的检测接口
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """所有检测器的抽象基类"""

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """加载模型文件
        Args:
            model_path: RKNN 模型文件路径
        Returns:
            bool: 是否加载成功
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45):
        """执行目标检测
        Args:
            image: RGB 图像 (H, W, 3)
            conf: 置信度阈值
            iou: NMS IOU 阈值
        Returns:
            (result_image, detections)
            result_image: 绘制了检测框的图像 (np.ndarray, RGB)
            detections: 检测结果列表，每项为 dict:
                {"bbox": [x1,y1,x2,y2], "confidence": float,
                 "class_id": int, "class_name": str}
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        pass

    @abstractmethod
    def unload(self):
        """释放模型资源"""
        pass
