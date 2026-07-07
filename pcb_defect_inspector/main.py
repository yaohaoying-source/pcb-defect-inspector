"""
YOLO COCO 目标检测系统 - 主入口
支持 Windows / Ubuntu / RK3588 三平台
"""
import sys
import os

# RK3588 无桌面时自动启用离屏渲染（必须在导入 PyQt5 前设置）
if sys.platform.startswith("linux"):
    if "DISPLAY" not in os.environ and not os.path.exists("/tmp/.X11-unix/X0"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        print("[INFO] 无桌面环境，已启用 Qt 离屏渲染 (offscreen)")

# 确保项目根目录在 sys.path 中
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# conda 环境可能找不到系统预装的 rknnlite/opencv，补上系统路径
_sys_pkg = "/usr/local/lib/python3.10/dist-packages"
if os.path.isdir(_sys_pkg) and _sys_pkg not in sys.path:
    sys.path.append(_sys_pkg)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow


def main():
    # 启用高分屏适配
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("PCB Defect Inspector")
    app.setApplicationDisplayName("PCB Defect Inspector")
    app.setOrganizationName("YOLODetect")

    # 全局样式
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QPushButton { padding: 5px 12px; font-size: 12px; }
        QLabel { font-size: 13px; }
        QGroupBox {
            font-weight: bold; font-size: 13px;
            margin-top: 6px; padding-top: 10px;
        }
        QTabBar::tab { padding: 5px 12px; min-width: 72px; }
        QStatusBar { font-size: 12px; }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
