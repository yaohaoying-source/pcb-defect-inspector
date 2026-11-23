import configparser
import importlib.util
from pathlib import Path

from PyQt5.QtCore import QSettings


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pcb_config", ROOT / "pcb_defect_inspector" / "utils" / "config.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_legacy_import_preserves_new_values(tmp_path, monkeypatch):
    def settings(org, app):
        return QSettings(str(tmp_path / org / (app + ".ini")), QSettings.IniFormat)

    class Factory:
        NoError = QSettings.NoError

        def __new__(cls, org, app):
            return settings(org, app)

    monkeypatch.setattr(module, "QSettings", Factory)
    old = settings("YOLODetect", "YOLO-COCO-Detect")
    old.setValue("server/host", "old.example.test")
    old.setValue("detection/confidence", "0.42")
    old.sync()
    current = module.AppConfig._settings()
    current.setValue("server/host", "new.example.test")
    current.sync()

    assert module.AppConfig.import_legacy_settings() == 1
    assert module.AppConfig.get_server_host() == "new.example.test"
    assert module.AppConfig.get_confidence() == 0.42
    assert module.AppConfig.import_legacy_settings() == 0


def test_example_fields_and_placeholder_ports(tmp_path, monkeypatch):
    example = ROOT / "deploy" / "YOLO-COCO-Detect.conf.example"
    data = configparser.RawConfigParser()
    data.read(example, encoding="utf-8")
    assert set(data.sections()) == {"deepseek", "detection", "model", "server"}
    assert data["deepseek"]["api_key"] == "YOUR_API_KEY_HERE"
    assert data["server"]["host"] == "127.0.0.1"
    settings = QSettings(str(example), QSettings.IniFormat)
    monkeypatch.setattr(module.AppConfig, "_settings", classmethod(lambda cls: settings))
    assert module.AppConfig.get_server_port() == 12345
    assert module.AppConfig.get_server_web_port() == 5000
    assert module.AppConfig.default_model_dir().endswith("models")


def test_settings_dialog_has_explicit_legacy_import(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app_root = str(ROOT / "pcb_defect_inspector")
    monkeypatch.syspath_prepend(app_root)
    from PyQt5.QtWidgets import QApplication, QPushButton
    from ui.settings_dialog import SettingsDialog
    from utils.config import AppConfig

    settings = QSettings(str(tmp_path / "isolated.ini"), QSettings.IniFormat)
    monkeypatch.setattr(AppConfig, "_settings", classmethod(lambda cls: settings))

    app = QApplication.instance() or QApplication([])
    dialog = SettingsDialog()
    labels = [button.text() for button in dialog.findChildren(QPushButton)]
    assert "导入旧版板端设置" in labels
    dialog.close()
