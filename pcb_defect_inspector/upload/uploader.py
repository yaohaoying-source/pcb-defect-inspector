"""
TCP 文件上传模块 - QThread 异步上传
"""
import os
import socket
import time

from .protocol import (
    send_file_meta, send_file_chunk, send_file_end,
    CHUNK_SIZE
)


def upload_file(filepath, host, port, progress_callback=None):
    """上传文件到 TCP 服务器

    Args:
        filepath: 要上传的文件路径
        host: 服务器 IP
        port: 服务器端口
        progress_callback: 进度回调函数，接收 int (0-100)

    Raises:
        ConnectionError: 连接失败
        OSError: 网络异常
        FileNotFoundError: 文件不存在
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)

    try:
        sock.connect((host, port))
        send_file_meta(sock, filename, filesize)

        sent_size = 0
        with open(filepath, "rb") as f:
            while sent_size < filesize:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                send_file_chunk(sock, chunk)
                sent_size += len(chunk)

                if progress_callback:
                    progress = int(sent_size / filesize * 100)
                    progress_callback(progress)

        send_file_end(sock)
    except socket.timeout:
        raise ConnectionError("连接超时，请检查服务器地址和端口")
    except ConnectionRefusedError:
        raise ConnectionError(f"无法连接到服务器 {host}:{port}，请确保服务端已启动")
    finally:
        sock.close()
