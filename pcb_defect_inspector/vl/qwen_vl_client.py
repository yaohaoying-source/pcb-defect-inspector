"""Qwen3-VL local demo subprocess wrapper."""
import os
import subprocess
import threading
import time


def normalize_session_question(question, first_turn):
    question = " ".join(question.strip().split())
    if first_turn and not question.startswith("<image>"):
        return "<image>" + question
    return question


class QwenVLClient:
    """Runs the RK3588 Qwen3-VL demo package as a subprocess."""

    VISION_MODEL = "qwen3-vl-2b_vision_rk3588.rknn"
    LLM_MODEL = "qwen3-vl-2b-instruct_w8a8_rk3588.rkllm"

    def __init__(self, demo_dir, timeout_sec=300):
        self.demo_dir = os.path.abspath(demo_dir)
        self.timeout_sec = timeout_sec

    def validate(self):
        required = [
            "demo",
            self.VISION_MODEL,
            self.LLM_MODEL,
            os.path.join("lib", "librkllmrt.so"),
            os.path.join("lib", "librknnrt.so"),
        ]
        missing = [
            item for item in required
            if not os.path.exists(os.path.join(self.demo_dir, item))
        ]
        if missing:
            raise FileNotFoundError("Qwen3-VL 运行目录缺少: " + ", ".join(missing))

    def analyze_image(self, image_path, prompt):
        self.validate()
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        cmd = [
            os.path.join(self.demo_dir, "demo"),
            image_path,
            self.VISION_MODEL,
            self.LLM_MODEL,
            "2048",
            "4096",
            "3",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
        ]
        env = os.environ.copy()
        lib_dir = os.path.join(self.demo_dir, "lib")
        env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

        # This demo expects visual questions to start with the image placeholder.
        prompt = " ".join(prompt.strip().split())
        if not prompt.startswith("<image>"):
            prompt = "<image>" + prompt
        input_text = prompt + "\nexit\n"
        output, returncode = self._run_with_pty(cmd, input_text, env)
        output = output.strip()
        if returncode != 0:
            raise RuntimeError(f"Qwen3-VL 运行失败(ret={returncode}):\n{output}")
        return self._extract_reply(output)

    def _run_with_pty(self, cmd, input_text, env):
        import pty
        import select

        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            cmd,
            cwd=self.demo_dir,
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)

        output_parts = []
        sent_prompt = False
        sent_exit = False
        start = time.time()

        try:
            while True:
                if time.time() - start > self.timeout_sec:
                    proc.kill()
                    raise TimeoutError(f"Qwen3-VL 分析超时({self.timeout_sec}s)")

                ready, _, _ = select.select([master_fd], [], [], 0.2)
                if ready:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="ignore")
                    output_parts.append(text)
                    joined = "".join(output_parts)

                    if not sent_prompt and "user:" in joined:
                        os.write(master_fd, input_text.splitlines()[0].encode("utf-8") + b"\n")
                        sent_prompt = True
                    elif sent_prompt and not sent_exit and joined.rstrip().endswith("user:"):
                        os.write(master_fd, b"exit\n")
                        sent_exit = True

                if proc.poll() is not None:
                    break
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass

        return "".join(output_parts), proc.wait()

    @staticmethod
    def _extract_reply(output):
        lines = []
        skip_prefixes = (
            "W rkllm:",
            "I rkllm:",
            "W RKNN:",
            "I RKNN:",
            "main:",
            "===",
            "****************",
            "[0]",
            "[1]",
            "output tensors",
            "index=",
            "model input",
            "input tensors",
            "rkllm init",
            "user:",
        )
        for raw in output.splitlines():
            line = raw.strip()
            if not line:
                continue
            lowered = line.lower()
            if any(line.startswith(prefix) or lowered.startswith(prefix.lower()) for prefix in skip_prefixes):
                continue
            if "cost" in lowered and "ms" in lowered:
                continue
            if "可输入以下问题" in line:
                continue
            if line.startswith("<image>") or line == "exit":
                continue
            if lowered.startswith("robot:"):
                raw = raw.split(":", 1)[1].lstrip()
            lines.append(raw)
        return "\n".join(lines).strip() or output


class QwenVLSession:
    """Persistent interactive Qwen3-VL demo session for one image."""

    VISION_MODEL = QwenVLClient.VISION_MODEL
    LLM_MODEL = QwenVLClient.LLM_MODEL

    def __init__(self, demo_dir, image_path, timeout_sec=300):
        self.demo_dir = os.path.abspath(demo_dir)
        self.image_path = image_path
        self.timeout_sec = timeout_sec
        self.proc = None
        self.master_fd = None
        self._lock = threading.Lock()
        self._first_turn = True
        self._closed = False

    def ask(self, question):
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen3-VL 会话已关闭")
            if self.proc is None:
                self._start()
            normalized = normalize_session_question(question, self._first_turn)
            self._write_line(normalized)
            answer = self._read_until_user_prompt()
            self._first_turn = False
            return QwenVLClient._extract_reply(answer)

    def close(self):
        with self._lock:
            self._closed = True
            if self.master_fd is not None:
                try:
                    self._write_line("exit")
                except Exception:
                    pass
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=2)
                except Exception:
                    self.proc.kill()
            if self.master_fd is not None:
                try:
                    os.close(self.master_fd)
                except OSError:
                    pass
            self.master_fd = None
            self.proc = None

    def _start(self):
        import pty

        QwenVLClient(self.demo_dir, self.timeout_sec).validate()
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"图片不存在: {self.image_path}")

        cmd = [
            os.path.join(self.demo_dir, "demo"),
            self.image_path,
            self.VISION_MODEL,
            self.LLM_MODEL,
            "2048",
            "4096",
            "3",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
        ]
        env = os.environ.copy()
        lib_dir = os.path.join(self.demo_dir, "lib")
        env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

        master_fd, slave_fd = pty.openpty()
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.demo_dir,
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        self.master_fd = master_fd
        self._read_until_user_prompt()

    def _write_line(self, text):
        os.write(self.master_fd, text.encode("utf-8") + b"\n")

    def _read_until_user_prompt(self):
        import select

        output_parts = []
        start = time.time()
        while True:
            if time.time() - start > self.timeout_sec:
                raise TimeoutError(f"Qwen3-VL 响应超时({self.timeout_sec}s)")
            if self.proc is not None and self.proc.poll() is not None:
                break

            ready, _, _ = select.select([self.master_fd], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(self.master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="ignore")
            output_parts.append(text)
            joined = "".join(output_parts)
            if joined.rstrip().endswith("user:"):
                break
        return "".join(output_parts)
