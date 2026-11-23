import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
APP_DIR = REPO_DIR / "pcb_defect_inspector"
TEST_DIR = None
MODEL_PATH = None
OUTPUT_DIR = None
UPLOAD_URL = None
CONF = 0.25
IOU = 0.45
UPLOAD_SAMPLE_LIMIT = 30

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short_circuit",
    "spur",
    "spurious_copper",
]


sys.path.insert(0, str(APP_DIR))
from detection.rknn_detector import RknnDetector  # noqa: E402
from detection.postprocess import process_yolo_output  # noqa: E402


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * pct / 100.0))
    return values[max(0, min(idx, len(values) - 1))]


def mean(values):
    return statistics.mean(values) if values else 0.0


def read_labels(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ids = []
            for item in row.get("expected_class_ids", "").split(";"):
                item = item.strip()
                if item:
                    ids.append(int(item))
            row["_expected_ids"] = sorted(set(ids))
            rows.append(row)
    return rows


def mem_info_mb():
    rss = 0
    vm = 0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
            elif line.startswith("VmSize:"):
                vm = int(line.split()[1]) / 1024
    except Exception:
        pass
    return rss, vm


def system_snapshot():
    load1, load5, load15 = os.getloadavg()
    rss, vm = mem_info_mb()
    mem_total = 0
    mem_avail = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) / 1024
            elif line.startswith("MemAvailable:"):
                mem_avail = int(line.split()[1]) / 1024
    except Exception:
        pass
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "load1": round(load1, 3),
        "load5": round(load5, 3),
        "load15": round(load15, 3),
        "process_rss_mb": round(rss, 1),
        "process_vmsize_mb": round(vm, 1),
        "mem_total_mb": round(mem_total, 1),
        "mem_available_mb": round(mem_avail, 1),
    }


def monitor(stop_event, out_csv):
    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "time",
            "load1",
            "load5",
            "load15",
            "process_rss_mb",
            "process_vmsize_mb",
            "mem_total_mb",
            "mem_available_mb",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        while not stop_event.is_set():
            writer.writerow(system_snapshot())
            file.flush()
            stop_event.wait(2.0)
        writer.writerow(system_snapshot())


def timed_detect(detector, image_rgb, conf, iou):
    h, w = image_rgb.shape[:2]
    timings = {}

    t0 = time.perf_counter()
    cur = detector._buf_idx
    cv2.resize(
        image_rgb,
        detector.input_size,
        dst=detector._infer_buf[cur][0],
        interpolation=cv2.INTER_NEAREST,
    )
    detector._buf_idx = 1 - cur
    t1 = time.perf_counter()

    outputs = detector.rknn.inference(inputs=[detector._infer_buf[cur]])
    t2 = time.perf_counter()

    if not detector._num_classes_known:
        detector.num_classes = detector._infer_num_classes(outputs)
        detector._num_classes_known = True

    raw_detections = process_yolo_output(
        outputs,
        conf_thres=conf,
        iou_thres=iou,
        input_size=detector.input_size,
        num_classes=detector.num_classes,
    )
    t3 = time.perf_counter()

    scale_x = w / detector.input_size[0]
    scale_y = h / detector.input_size[1]
    detections = []
    for det in raw_detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = int(det["class_id"])
        detections.append(
            {
                "bbox": [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                ],
                "confidence": float(det["confidence"]),
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
            }
        )
    t4 = time.perf_counter()

    timings["preprocess_ms"] = (t1 - t0) * 1000
    timings["infer_ms"] = (t2 - t1) * 1000
    timings["postprocess_ms"] = (t3 - t2) * 1000
    timings["scale_ms"] = (t4 - t3) * 1000
    return detections, timings


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_upload(result_dir, upload_rows):
    try:
        import requests
    except Exception as exc:
        for row in upload_rows:
            row["upload_success"] = False
            row["upload_ms"] = ""
            row["upload_status"] = f"requests import failed: {exc}"
        return upload_rows

    url = UPLOAD_URL
    for row in upload_rows:
        path = result_dir / "detect_images" / row["result_image"]
        t0 = time.perf_counter()
        try:
            with path.open("rb") as file:
                resp = requests.post(url, files={"file": (path.name, file, "image/jpeg")}, timeout=20)
            row["upload_success"] = resp.ok
            row["upload_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            row["upload_status"] = str(resp.status_code)
        except Exception as exc:
            row["upload_success"] = False
            row["upload_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            row["upload_status"] = str(exc)
    return upload_rows


def main():
    global TEST_DIR, MODEL_PATH, OUTPUT_DIR, UPLOAD_URL
    parser = argparse.ArgumentParser(description="Run the PCB image-level benchmark")
    parser.add_argument("--dataset", type=Path, required=True, help="Directory with labels.csv and images/")
    parser.add_argument("--model", type=Path, required=True, help="RKNN model path")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "outputs" / "benchmarks")
    parser.add_argument("--http-upload-url", help="Optional upload endpoint; omitted by default")
    args = parser.parse_args()
    TEST_DIR, MODEL_PATH, OUTPUT_DIR, UPLOAD_URL = args.dataset, args.model, args.output_dir, args.http_upload_url
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = OUTPUT_DIR / f"run_{timestamp}"
    detect_dir = result_dir / "detect_images"
    result_dir.mkdir(parents=True, exist_ok=True)
    detect_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor,
        args=(stop_event, result_dir / "system_monitor.csv"),
        daemon=True,
    )
    monitor_thread.start()

    labels = read_labels(TEST_DIR / "labels.csv")
    detector = RknnDetector()
    load_t0 = time.perf_counter()
    if not detector.load_model(str(MODEL_PATH)):
        raise RuntimeError(f"load model failed: {MODEL_PATH}")
    load_ms = (time.perf_counter() - load_t0) * 1000

    rows = []
    start = time.perf_counter()
    for idx, label in enumerate(labels, 1):
        image_path = TEST_DIR / "images" / label["image"]
        read_t0 = time.perf_counter()
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        read_ms = (time.perf_counter() - read_t0) * 1000
        if bgr is None:
            rows.append(
                {
                    "image": label["image"],
                    "expected_classes": label["expected_classes"],
                    "error": "cv2.imread failed",
                }
            )
            continue

        rgb_t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_ms = (time.perf_counter() - rgb_t0) * 1000

        det_t0 = time.perf_counter()
        detections, timings = timed_detect(detector, rgb, CONF, IOU)
        detect_no_draw_ms = (time.perf_counter() - det_t0) * 1000

        draw_t0 = time.perf_counter()
        drawn_rgb = detector.draw_detections(rgb, detections, copy_image=True)
        result_image = f"det_{label['image']}"
        cv2.imwrite(str(detect_dir / result_image), cv2.cvtColor(drawn_rgb, cv2.COLOR_RGB2BGR))
        draw_save_ms = (time.perf_counter() - draw_t0) * 1000

        expected_ids = set(label["_expected_ids"])
        pred_ids = [int(item["class_id"]) for item in detections]
        pred_names = [item["class_name"] for item in detections]
        best = max(detections, key=lambda item: item.get("confidence", 0.0), default=None)
        class_hit = bool(expected_ids.intersection(pred_ids))
        top1_hit = bool(best and int(best["class_id"]) in expected_ids)

        rows.append(
            {
                "index": idx,
                "image": label["image"],
                "expected_classes": label["expected_classes"],
                "expected_class_ids": label["expected_class_ids"],
                "object_count_gt": label["object_count"],
                "pred_count": len(detections),
                "pred_classes": ";".join(sorted(set(pred_names))),
                "pred_class_ids": ";".join(str(i) for i in sorted(set(pred_ids))),
                "best_class": best["class_name"] if best else "",
                "best_confidence": round(float(best["confidence"]), 6) if best else "",
                "class_hit": class_hit,
                "top1_hit": top1_hit,
                "read_ms": round(read_ms, 3),
                "rgb_convert_ms": round(rgb_ms, 3),
                "preprocess_ms": round(timings["preprocess_ms"], 3),
                "infer_ms": round(timings["infer_ms"], 3),
                "postprocess_ms": round(timings["postprocess_ms"], 3),
                "scale_ms": round(timings["scale_ms"], 3),
                "detect_no_draw_ms": round(detect_no_draw_ms, 3),
                "draw_save_ms": round(draw_save_ms, 3),
                "total_ms": round(read_ms + rgb_ms + detect_no_draw_ms + draw_save_ms, 3),
                "result_image": result_image,
                "error": "",
            }
        )

        if idx % 25 == 0:
            print(f"[batch] {idx}/{len(labels)} done")

    total_wall_s = time.perf_counter() - start
    detector.unload()
    stop_event.set()
    monitor_thread.join(timeout=5)

    fieldnames = [
        "index",
        "image",
        "expected_classes",
        "expected_class_ids",
        "object_count_gt",
        "pred_count",
        "pred_classes",
        "pred_class_ids",
        "best_class",
        "best_confidence",
        "class_hit",
        "top1_hit",
        "read_ms",
        "rgb_convert_ms",
        "preprocess_ms",
        "infer_ms",
        "postprocess_ms",
        "scale_ms",
        "detect_no_draw_ms",
        "draw_save_ms",
        "total_ms",
        "result_image",
        "error",
    ]
    write_csv(result_dir / "batch_detection_result.csv", rows, fieldnames)

    valid_rows = [row for row in rows if not row.get("error")]
    per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        expected = [row for row in valid_rows if str(cls_id) in str(row["expected_class_ids"]).split(";")]
        hit = [row for row in expected if str(row["class_hit"]).lower() == "true"]
        top1 = [row for row in expected if str(row["top1_hit"]).lower() == "true"]
        per_class[cls_name] = {
            "expected_images": len(expected),
            "class_hit": len(hit),
            "class_hit_rate": round(len(hit) / len(expected), 4) if expected else 0,
            "top1_hit": len(top1),
            "top1_hit_rate": round(len(top1) / len(expected), 4) if expected else 0,
        }

    timing_keys = [
        "read_ms",
        "rgb_convert_ms",
        "preprocess_ms",
        "infer_ms",
        "postprocess_ms",
        "detect_no_draw_ms",
        "draw_save_ms",
        "total_ms",
    ]
    timing_summary = {}
    for key in timing_keys:
        values = [float(row[key]) for row in valid_rows if row.get(key) not in ("", None)]
        timing_summary[key] = {
            "avg": round(mean(values), 3),
            "p50": round(percentile(values, 50), 3),
            "p95": round(percentile(values, 95), 3),
            "p99": round(percentile(values, 99), 3),
            "max": round(max(values), 3) if values else 0,
        }

    pred_counter = Counter()
    for row in valid_rows:
        for cls_name in str(row["pred_classes"]).split(";"):
            if cls_name:
                pred_counter[cls_name] += 1

    upload_candidates = valid_rows[:UPLOAD_SAMPLE_LIMIT] if UPLOAD_URL else []
    upload_rows = [
        {
            "image": row["image"],
            "result_image": row["result_image"],
            "bytes": (detect_dir / row["result_image"]).stat().st_size,
        }
        for row in upload_candidates
    ]
    if UPLOAD_URL:
        upload_rows = try_upload(result_dir, upload_rows)
    write_csv(
        result_dir / "upload_result.csv",
        upload_rows,
        ["image", "result_image", "bytes", "upload_success", "upload_ms", "upload_status"],
    )

    summary = {
        "test_dir": str(TEST_DIR),
        "result_dir": str(result_dir),
        "model_path": str(MODEL_PATH),
        "confidence": CONF,
        "iou": IOU,
        "image_count": len(labels),
        "valid_image_count": len(valid_rows),
        "load_model_ms": round(load_ms, 3),
        "total_wall_s": round(total_wall_s, 3),
        "throughput_images_per_s": round(len(valid_rows) / total_wall_s, 3) if total_wall_s else 0,
        "class_hit_count": sum(1 for row in valid_rows if str(row["class_hit"]).lower() == "true"),
        "class_hit_rate": round(
            sum(1 for row in valid_rows if str(row["class_hit"]).lower() == "true") / len(valid_rows),
            4,
        )
        if valid_rows
        else 0,
        "top1_hit_count": sum(1 for row in valid_rows if str(row["top1_hit"]).lower() == "true"),
        "top1_hit_rate": round(
            sum(1 for row in valid_rows if str(row["top1_hit"]).lower() == "true") / len(valid_rows),
            4,
        )
        if valid_rows
        else 0,
        "no_detection_count": sum(1 for row in valid_rows if int(row["pred_count"]) == 0),
        "predicted_class_image_counts": dict(pred_counter),
        "per_class": per_class,
        "timing_summary": timing_summary,
        "upload_success_count": sum(1 for row in upload_rows if row.get("upload_success") is True),
        "upload_count": len(upload_rows),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"RESULT_DIR={result_dir}")


if __name__ == "__main__":
    main()
