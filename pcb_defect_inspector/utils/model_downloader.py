"""
Download RKNN model files from the server web API.
"""
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def safe_model_filename(filename):
    name = os.path.basename((filename or "").strip())
    if not name or not name.lower().endswith(".rknn"):
        raise ValueError("请选择 .rknn 模型文件")
    return name


def build_server_base_url(host, web_port=5000):
    host = (host or "").strip()
    if not host:
        raise ValueError("服务器地址不能为空")
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}:{int(web_port)}"


def list_remote_models(host, web_port=5000, timeout=10):
    url = build_server_base_url(host, web_port) + "/api/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    models = json.loads(data)
    return [m for m in models if str(m.get("filename", "")).lower().endswith(".rknn")]


def download_remote_model(
    host,
    filename,
    target_dir=None,
    web_port=5000,
    timeout=300,
    progress_callback=None,
):
    if target_dir is None:
        target_dir = str(Path(__file__).resolve().parents[1] / "models")
    name = safe_model_filename(filename)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, name)
    part_path = target_path + ".part"

    quoted_name = urllib.parse.quote(name)
    url = build_server_base_url(host, web_port) + f"/api/models/download/{quoted_name}"

    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(part_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress_callback and total:
                    progress_callback(min(100, int(done * 100 / total)))

    os.replace(part_path, target_path)
    if progress_callback:
        progress_callback(100)
    return target_path
