"""Prompt construction for Qwen3-VL PCB defect conversations."""


def format_detections(detections):
    if not detections:
        return "检测模型没有给出明确缺陷框，请根据图片内容谨慎判断。"

    rows = []
    for idx, det in enumerate(detections, start=1):
        rows.append(
            f"{idx}. 类别={det.get('class_name', det.get('class_id'))}, "
            f"置信度={det.get('confidence', 0):.2f}, "
            f"位置={det.get('bbox')}"
        )
    return " ".join(rows)


def build_qwen_prompt(detections, history, question):
    history_rows = []
    for role, content in history[-6:]:
        label = "用户" if role == "user" else "助手"
        history_rows.append(f"{label}: {content}")

    history_text = " ".join(history_rows) if history_rows else "无"
    detection_text = format_detections(detections)

    return (
        "<image>"
        "你是PCB制造质量工程师。请结合当前图片、检测模型结果和对话历史回答用户问题。"
        "回答请用中文，结构清晰，优先给出可执行的生产建议；"
        "不要编造图中看不到的信息。 "
        f"检测模型结果: {detection_text} "
        f"对话历史: {history_text} "
        f"当前问题: {question.strip()}"
    )
