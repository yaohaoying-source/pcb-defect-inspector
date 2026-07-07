"""
主窗口 - 包含检测和聊天两个标签页的 QMainWindow
"""
import os

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu,
    QAction, QStatusBar, QMessageBox, QFileDialog, QDesktopWidget
)
from PyQt5.QtCore import Qt

from .detection_tab import DetectionTab
from .camera_tab import CameraTab
from .chat_tab import ChatTab
from .settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """应用主窗口，包含检测和聊天两个功能标签页"""

    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.setWindowTitle("PCB Defect Inspector")
        self.setMinimumSize(760, 460)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self._screen_fit_done = False

        self._init_ui()
        self._apply_background_style()
        self._init_menu()
        self.fit_to_screen()

    def _init_ui(self):
        """初始化中心控件（标签页）"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("MainTabs")
        self.tab_widget.setTabPosition(QTabWidget.North)

        self.detection_tab = DetectionTab(self)
        self.camera_tab = CameraTab(self)
        self.chat_tab = ChatTab(self)
        self.detection_tab.setObjectName("DetectionTab")
        self.camera_tab.setObjectName("CameraTab")
        self.chat_tab.setObjectName("ChatTab")

        self.tab_widget.addTab(self.detection_tab, "图片检测")
        self.tab_widget.addTab(self.camera_tab, "实时检测")
        self.tab_widget.addTab(self.chat_tab, "DeepSeek 聊天")

        self.setCentralWidget(self.tab_widget)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def _apply_background_style(self):
        """Apply the product background image without reducing control readability."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bg_path = os.path.join(base_dir, "assets", "app_background.jpg").replace("\\", "/")
        self.setStyleSheet(f"""
            QMainWindow#MainWindow {{
                background-image: url("{bg_path}");
                background-position: center;
                background-repeat: no-repeat;
                background-color: #ffffff;
            }}
            QTabWidget#MainTabs {{
                background: transparent;
            }}
            QTabWidget#MainTabs::pane {{
                border: 1px solid rgba(30, 64, 96, 80);
                background: rgba(255, 255, 255, 218);
            }}
            QTabWidget#MainTabs QStackedWidget {{
                background: rgba(255, 255, 255, 218);
            }}
            QWidget#DetectionTab,
            QWidget#CameraTab,
            QWidget#ChatTab {{
                background: rgba(255, 255, 255, 218);
            }}
            QTabBar::tab {{
                background: rgba(241, 245, 249, 235);
                color: #143b5a;
                padding: 5px 12px;
                min-width: 72px;
            }}
            QTabBar::tab:selected {{
                background: rgba(255, 255, 255, 245);
                color: #0f3554;
                font-weight: 600;
            }}
            QMenuBar,
            QStatusBar {{
                background: rgba(255, 255, 255, 230);
            }}
            QGroupBox {{
                background: rgba(255, 255, 255, 225);
                border: 1px solid rgba(30, 64, 96, 60);
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """)

    def _init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()

        # ---- 文件菜单 ----
        file_menu = menubar.addMenu("文件(&F)")

        open_img_action = QAction("打开图片(&O)...", self)
        open_img_action.setShortcut("Ctrl+O")
        open_img_action.triggered.connect(self.detection_tab.open_image)
        file_menu.addAction(open_img_action)

        open_model_action = QAction("加载模型(&M)...", self)
        open_model_action.setShortcut("Ctrl+M")
        open_model_action.triggered.connect(self.detection_tab.load_model_dialog)
        file_menu.addAction(open_model_action)

        file_menu.addSeparator()

        exit_action = QAction("退出(&Q)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ---- 设置菜单 ----
        settings_menu = menubar.addMenu("设置(&S)")

        config_action = QAction("应用设置(&C)...", self)
        config_action.setShortcut("Ctrl+,")
        config_action.triggered.connect(self._open_settings)
        settings_menu.addAction(config_action)

        view_menu = menubar.addMenu("视图(&V)")

        fit_action = QAction("适配当前屏幕(&F)", self)
        fit_action.setShortcut("F11")
        fit_action.triggered.connect(self.fit_to_screen)
        view_menu.addAction(fit_action)

        max_action = QAction("最大化/还原(&X)", self)
        max_action.setShortcut("Ctrl+F11")
        max_action.triggered.connect(self.toggle_maximized)
        view_menu.addAction(max_action)

        # ---- 帮助菜单 ----
        help_menu = menubar.addMenu("帮助(&H)")

        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self)
        if dialog.exec_():
            # 如果有需要实时更新的配置，在这里处理
            self.status_bar.showMessage("设置已保存", 3000)

    def _show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "RK3588 COCO 目标检测 v2.0\n\n"
            "基于 PyQt5 + RKNN NPU 加速\n"
            "支持图片检测 / 摄像头实时推理\n"
            "支持 TCP/IP 图片上传和 DeepSeek 聊天"
        )

    def fit_to_screen(self):
        """Resize and place the window inside the current available screen."""
        screen = QDesktopWidget().availableGeometry(self)
        width_ratio = 0.96 if screen.width() <= 1280 else 0.86
        height_ratio = 0.92 if screen.height() <= 720 else 0.86
        w = max(760, min(int(screen.width() * width_ratio), 1400))
        h = max(460, min(int(screen.height() * height_ratio), 900))
        self.showNormal()
        self.resize(w, h)
        x = screen.x() + max(0, (screen.width() - w) // 2)
        y = screen.y() + max(0, (screen.height() - h) // 2)
        self.move(x, y)

    def toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
            self.fit_to_screen()
        else:
            self.showMaximized()

    def closeEvent(self, event):
        """关闭窗口前清理资源"""
        self.detection_tab.cleanup()
        self.camera_tab.cleanup()
        super().closeEvent(event)
