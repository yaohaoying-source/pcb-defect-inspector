from pathlib import Path

path = Path(__file__).resolve().parents[1] / "pcb_defect_inspector/ui/camera_tab.py"
text = path.read_text(encoding="utf-8")

required = ["摄像头FPS", "推理FPS", "显示FPS"]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit("missing FPS labels: " + ", ".join(missing))

if "camera_fps" not in text:
    raise SystemExit("missing separate camera_fps data path")

print("camera fps label check passed")
