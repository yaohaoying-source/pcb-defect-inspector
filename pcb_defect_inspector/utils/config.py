"""
配置管理模块 - 使用 QSettings 持久化应用配置
"""
from pathlib import Path

from PyQt5.QtCore import QSettings


class AppConfig:
    """应用配置管理，基于 QSettings 实现跨平台持久化"""

    _APP_NAME = "PCB-Defect-Inspector"
    _ORG_NAME = "PCBDefectInspector"

    @classmethod
    def _settings(cls) -> QSettings:
        return QSettings(cls._ORG_NAME, cls._APP_NAME)

    @classmethod
    def import_legacy_settings(cls) -> int:
        """Import missing board settings only after the user requests it."""
        old = QSettings("YOLODetect", "YOLO-COCO-Detect")
        current = cls._settings()
        imported = 0
        for key in old.allKeys():
            if not current.contains(key):
                current.setValue(key, old.value(key))
                imported += 1
        current.sync()
        if current.status() != QSettings.NoError:
            raise OSError("无法写入本机设置")
        return imported

    @staticmethod
    def default_model_dir() -> str:
        return str(Path(__file__).resolve().parents[1] / "models")

    @staticmethod
    def _port_or_default(value, default: int) -> int:
        try:
            port = int(value)
        except (TypeError, ValueError):
            return default
        return port if 1 <= port <= 65535 else default

    # ---- 服务器配置 ----
    @classmethod
    def get_server_host(cls) -> str:
        return cls._settings().value("server/host", "127.0.0.1")

    @classmethod
    def set_server_host(cls, host: str):
        cls._settings().setValue("server/host", host)

    @classmethod
    def get_server_port(cls) -> int:
        val = cls._settings().value("server/port", "12345")
        return cls._port_or_default(val, 12345)

    @classmethod
    def set_server_port(cls, port: int):
        cls._settings().setValue("server/port", str(port))

    @classmethod
    def get_server_web_port(cls) -> int:
        val = cls._settings().value("server/web_port", "5000")
        return cls._port_or_default(val, 5000)

    @classmethod
    def set_server_web_port(cls, port: int):
        cls._settings().setValue("server/web_port", str(port))

    @classmethod
    def get_model_download_dir(cls) -> str:
        return cls._settings().value("model/download_dir", cls.default_model_dir())

    @classmethod
    def set_model_download_dir(cls, path: str):
        cls._settings().setValue("model/download_dir", path)

    # ---- DeepSeek 配置 ----
    @classmethod
    def get_deepseek_api_key(cls) -> str:
        return cls._settings().value("deepseek/api_key", "")

    @classmethod
    def set_deepseek_api_key(cls, key: str):
        cls._settings().setValue("deepseek/api_key", key)

    @classmethod
    def get_deepseek_model(cls) -> str:
        return cls._settings().value("deepseek/model", "deepseek-chat")

    @classmethod
    def set_deepseek_model(cls, model: str):
        cls._settings().setValue("deepseek/model", model)

    # ---- 检测配置 ----
    @classmethod
    def get_confidence(cls) -> float:
        val = cls._settings().value("detection/confidence", "0.25")
        return float(val) if val else 0.25

    @classmethod
    def set_confidence(cls, conf: float):
        cls._settings().setValue("detection/confidence", str(conf))

    @classmethod
    def get_iou(cls) -> float:
        val = cls._settings().value("detection/iou", "0.45")
        return float(val) if val else 0.45

    @classmethod
    def set_iou(cls, iou: float):
        cls._settings().setValue("detection/iou", str(iou))

    @classmethod
    def get_last_model_path(cls) -> str:
        return cls._settings().value("detection/last_model", "")

    @classmethod
    def set_last_model_path(cls, path: str):
        cls._settings().setValue("detection/last_model", path)

    # ---- Qwen3-VL 本地配置 ----
    @classmethod
    def get_qwen_vl_dir(cls) -> str:
        return cls._settings().value("qwen_vl/demo_dir", "")

    @classmethod
    def set_qwen_vl_dir(cls, path: str):
        cls._settings().setValue("qwen_vl/demo_dir", path)

    @classmethod
    def get_qwen_vl_timeout(cls) -> int:
        val = cls._settings().value("qwen_vl/timeout_sec", "300")
        return int(val) if val else 300

    @classmethod
    def set_qwen_vl_timeout(cls, timeout_sec: int):
        cls._settings().setValue("qwen_vl/timeout_sec", str(timeout_sec))

