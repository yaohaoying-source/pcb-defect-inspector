"""
TCP通信协议定义
消息帧格式: [4字节消息类型][4字节数据长度][数据]
"""
import struct

# 消息类型常量
MSG_CHAT = 0x01       # 聊天文本
MSG_FILE_META = 0x02  # 文件元信息 (filename|filesize)
MSG_FILE_DATA = 0x03  # 文件数据块
MSG_FILE_END = 0x04   # 文件传输结束
MSG_ERROR = 0xFF      # 错误/退出

HEADER_SIZE = 8       # 消息头大小 (type 4B + length 4B)
CHUNK_SIZE = 65536    # 64KB 数据块


def recv_exact(sock, size):
    """精确接收指定字节数的数据"""
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("连接已断开")
        buf += chunk
    return buf


def send_message(sock, msg_type, data=b""):
    """发送一条完整消息"""
    header = struct.pack("!II", msg_type, len(data))
    sock.sendall(header + data)


def recv_message(sock):
    """接收一条完整消息，返回 (msg_type, data)"""
    header = recv_exact(sock, HEADER_SIZE)
    msg_type, data_len = struct.unpack("!II", header)
    data = recv_exact(sock, data_len) if data_len > 0 else b""
    return msg_type, data


def send_chat(sock, text):
    """发送聊天消息"""
    send_message(sock, MSG_CHAT, text.encode("utf-8"))


def recv_chat(sock):
    """接收聊天消息，返回文本"""
    msg_type, data = recv_message(sock)
    if msg_type == MSG_ERROR:
        return None
    return data.decode("utf-8")


def send_file_meta(sock, filename, filesize):
    """发送文件元信息"""
    meta = f"{filename}|{filesize}".encode("utf-8")
    send_message(sock, MSG_FILE_META, meta)


def recv_file_meta(sock):
    """接收文件元信息，返回 (filename, filesize)"""
    msg_type, data = recv_message(sock)
    if msg_type != MSG_FILE_META:
        raise ValueError(f"预期文件元信息，收到类型 {msg_type}")
    meta = data.decode("utf-8")
    filename, filesize = meta.split("|", 1)
    return filename, int(filesize)


def send_file_chunk(sock, chunk):
    """发送一块文件数据"""
    send_message(sock, MSG_FILE_DATA, chunk)


def send_file_end(sock):
    """发送文件传输结束标记"""
    send_message(sock, MSG_FILE_END)


def send_error(sock, text="错误"):
    """发送错误消息"""
    send_message(sock, MSG_ERROR, text.encode("utf-8"))
