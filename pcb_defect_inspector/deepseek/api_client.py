"""
DeepSeek API 客户端 - 调用 OpenAI 兼容接口
"""
import requests


class DeepSeekClient:
    """DeepSeek API 客户端，基于 OpenAI 兼容接口"""

    BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def send_message(self, messages: list) -> str:
        """发送消息并获取回复
        Args:
            messages: 消息列表 [{"role": str, "content": str}, ...]
        Returns:
            模型回复文本
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": 4096,
        }

        try:
            resp = self.session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 401:
                raise RuntimeError("API Key 无效，请在设置中检查")
            elif status == 429:
                raise RuntimeError("请求过于频繁，请稍后重试")
            elif status == 402:
                raise RuntimeError("账户余额不足")
            else:
                detail = e.response.text[:200] if e.response.text else ""
                raise RuntimeError(f"API 请求失败 (HTTP {status}): {detail}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("无法连接 DeepSeek API，请检查网络")
        except requests.exceptions.Timeout:
            raise RuntimeError("DeepSeek API 请求超时")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"API 响应格式异常: {e}")
