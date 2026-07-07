"""Helpers for storing generated detection result images."""
import os
import re
import time


def _safe_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._-")
    return stem or "image"


def build_result_image_path(source_path, timestamp=None, base_dir=None):
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    base_dir = base_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "detect_results")
    os.makedirs(base_dir, exist_ok=True)
    filename = f"{_safe_stem(source_path)}_detect_{timestamp}.jpg"
    return os.path.join(base_dir, filename)
