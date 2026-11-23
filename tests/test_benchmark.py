import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_batch_is_offline_by_default(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "pcb_batch", ROOT / "scripts" / "pcb_300_batch_test.py"
    )
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    assert cv2.imwrite(str(dataset / "images" / "sample.png"), np.zeros((8, 8, 3), np.uint8))
    with (dataset / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "expected_classes", "expected_class_ids", "object_count"])
        writer.writeheader()
        writer.writerow({"image": "sample.png", "expected_classes": "missing_hole", "expected_class_ids": "0", "object_count": "1"})

    class Detector:
        def load_model(self, path):
            return True

        def draw_detections(self, image, detections, copy_image):
            return image.copy()

        def unload(self):
            pass

    def detect(detector, image, conf, iou):
        detection = {"class_id": 0, "class_name": "missing_hole", "confidence": 0.9}
        return [detection], {key: 1 for key in ("preprocess_ms", "infer_ms", "postprocess_ms", "scale_ms")}

    def no_upload(*args):
        raise AssertionError("the default run must not upload")

    monkeypatch.setattr(batch, "RknnDetector", Detector)
    monkeypatch.setattr(batch, "timed_detect", detect)
    monkeypatch.setattr(batch, "try_upload", no_upload)
    monkeypatch.setattr(batch, "system_snapshot", lambda: {
        "time": "", "load1": 0, "load5": 0, "load15": 0,
        "process_rss_mb": 0, "process_vmsize_mb": 0, "mem_total_mb": 0, "mem_available_mb": 0,
    })
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "batch", "--dataset", str(dataset), "--model", str(tmp_path / "model.rknn"),
        "--output-dir", str(output),
    ])
    batch.main()
    result = json.loads(next(output.glob("run_*/summary.json")).read_text(encoding="utf-8"))
    assert result["image_count"] == result["class_hit_count"] == result["top1_hit_count"] == 1
    assert result["upload_count"] == 0
