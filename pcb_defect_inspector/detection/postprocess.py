"""
YOLO 后处理优化版

针对当前 RKNN 多输出结构：
(reg, cls, obj) * 3，其中 reg 为 DFL 分布输出。

优化点：
1. DFL softmax 只对超过置信度阈值的候选点计算，不再对全部网格点计算。
2. 缓存 grid center，避免每帧重复 meshgrid。
3. 使用 xyxy 格式的 NumPy NMS，避免 cv2.dnn.NMSBoxes 的列表转换开销。
4. NMS 默认按类别执行，避免不同缺陷类别互相抑制。
"""
import numpy as np

_GRID_CACHE = {}
_PROJECT_CACHE = {}
MAX_CANDIDATES_PER_SCALE = 1000


def process_yolo_output(outputs, conf_thres=0.25, iou_thres=0.45,
                        input_size=(640, 640), num_classes=80):
    """统一的 YOLO 输出后处理入口。"""
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 1:
            out = np.asarray(outputs[0], dtype=np.float32)
            return _process_single(out, conf_thres, iou_thres, None)
        return _process_multiscale(outputs, conf_thres, iou_thres, input_size)

    out = np.asarray(outputs, dtype=np.float32)
    return _process_single(out, conf_thres, iou_thres, num_classes)


def _process_single(out, conf_thres, iou_thres, num_classes):
    """单输出格式后处理，支持 [1,C,N] / [N,C]。"""
    if out.ndim == 3:
        if out.shape[0] == 1:
            out = out[0]
        if out.shape[0] < out.shape[1]:
            out = out.transpose(1, 0)
    elif out.ndim == 4 and out.shape[0] == 1:
        out = out.reshape(out.shape[1], -1).transpose(1, 0)

    if out.ndim != 2 or out.shape[1] <= 4:
        return []

    if num_classes is None or out.shape[1] != 4 + num_classes:
        num_classes = out.shape[1] - 4

    cls = out[:, 4:]
    class_ids = np.argmax(cls, axis=1).astype(np.int32)
    scores = cls[np.arange(len(out)), class_ids].astype(np.float32)

    keep = scores >= conf_thres
    if not np.any(keep):
        return []

    arr = out[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # cxcywh -> xyxy
    boxes = np.empty((arr.shape[0], 4), dtype=np.float32)
    boxes[:, 0] = arr[:, 0] - arr[:, 2] * 0.5
    boxes[:, 1] = arr[:, 1] - arr[:, 3] * 0.5
    boxes[:, 2] = arr[:, 0] + arr[:, 2] * 0.5
    boxes[:, 3] = arr[:, 1] + arr[:, 3] * 0.5

    keep_idx = _nms_xyxy_class_aware(boxes, scores, class_ids, iou_thres)
    return _pack_results(boxes, scores, class_ids, keep_idx)


def _process_multiscale(outputs, conf_thres, iou_thres, input_size):
    """处理 YOLOv8/YOLO11 RKNN 多尺度输出: (reg, cls, obj) * 3。"""
    all_boxes = []
    all_scores = []
    all_cls = []
    img_w, img_h = input_size

    i = 0
    n = len(outputs)
    while i + 2 < n:
        a_reg = np.asarray(outputs[i], dtype=np.float32)
        a_cls = np.asarray(outputs[i + 1], dtype=np.float32)
        a_obj = np.asarray(outputs[i + 2], dtype=np.float32)
        i += 3

        if a_reg.ndim != 4 or a_cls.ndim != 4:
            continue

        _, reg_ch, H, W = a_reg.shape
        if reg_ch % 4 != 0:
            continue

        cls_map = a_cls[0]                    # [C, H, W]
        cls_flat = cls_map.reshape(cls_map.shape[0], -1)
        class_ids = np.argmax(cls_flat, axis=0).astype(np.int32)
        class_scores = cls_flat[class_ids, np.arange(cls_flat.shape[1])]

        if a_obj.ndim == 4 and a_obj.shape[1] == 1:
            obj_scores = a_obj[0, 0].reshape(-1)
            scores = (class_scores * obj_scores).astype(np.float32)
        else:
            scores = class_scores.astype(np.float32)

        idx = np.flatnonzero(scores >= conf_thres)
        if idx.size == 0:
            continue

        # 防止低阈值场景下候选数过大拖慢 DFL/NMS。保留每个尺度最高分候选。
        if idx.size > MAX_CANDIDATES_PER_SCALE:
            local = np.argpartition(scores[idx], -MAX_CANDIDATES_PER_SCALE)[-MAX_CANDIDATES_PER_SCALE:]
            idx = idx[local]

        reg_max = reg_ch // 4
        project = _get_project(reg_max)

        # 关键优化：只对 keep 后的候选点计算 DFL softmax。
        # 原逻辑对 80*80 + 40*40 + 20*20 全部点计算，实时场景浪费较大。
        reg = a_reg[0].reshape(4, reg_max, -1)[:, :, idx]   # [4, reg_max, K]
        dist = (_softmax(reg, axis=1) * project.reshape(1, reg_max, 1)).sum(axis=1)

        cx, cy, stride_x, stride_y = _get_grid(H, W, img_w, img_h)
        cx_i = cx[idx]
        cy_i = cy[idx]

        boxes = np.empty((idx.size, 4), dtype=np.float32)
        boxes[:, 0] = cx_i - dist[0] * stride_x
        boxes[:, 1] = cy_i - dist[1] * stride_y
        boxes[:, 2] = cx_i + dist[2] * stride_x
        boxes[:, 3] = cy_i + dist[3] * stride_y

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h - 1)

        all_boxes.append(boxes)
        all_scores.append(scores[idx])
        all_cls.append(class_ids[idx])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_cls, axis=0)

    keep_idx = _nms_xyxy_class_aware(boxes, scores, class_ids, iou_thres)
    return _pack_results(boxes, scores, class_ids, keep_idx)


def _softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _get_project(reg_max):
    arr = _PROJECT_CACHE.get(reg_max)
    if arr is None:
        arr = np.arange(reg_max, dtype=np.float32)
        _PROJECT_CACHE[reg_max] = arr
    return arr


def _get_grid(H, W, img_w, img_h):
    key = (H, W, img_w, img_h)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    stride_x = img_w / float(W)
    stride_y = img_h / float(H)
    grid_y, grid_x = np.meshgrid(np.arange(H, dtype=np.float32),
                                 np.arange(W, dtype=np.float32),
                                 indexing="ij")
    cx = (grid_x.reshape(-1) + 0.5) * stride_x
    cy = (grid_y.reshape(-1) + 0.5) * stride_y
    cached = (cx, cy, stride_x, stride_y)
    _GRID_CACHE[key] = cached
    return cached


def _nms_xyxy_class_aware(boxes, scores, class_ids, iou_thres):
    """基于 xyxy 的类别感知 NMS。返回保留的全局索引。"""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)

    keep_all = []
    for cid in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == cid)
        keep_local = _nms_xyxy(boxes[idx], scores[idx], iou_thres)
        if keep_local.size:
            keep_all.append(idx[keep_local])

    if not keep_all:
        return np.empty((0,), dtype=np.int32)

    keep = np.concatenate(keep_all)
    # 输出按置信度降序，便于 UI 显示和后续截断。
    return keep[np.argsort(scores[keep])[::-1]]


def _nms_xyxy(boxes, scores, iou_thres):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= iou_thres]

    return np.asarray(keep, dtype=np.int32)


def _pack_results(boxes, scores, class_ids, keep_idx):
    if keep_idx is None or len(keep_idx) == 0:
        return []

    out = []
    for i in keep_idx:
        x1, y1, x2, y2 = boxes[i]
        out.append({
            "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "confidence": float(scores[i]),
            "class_id": int(class_ids[i]),
        })
    return out
