"""
IOU 目标跟踪器 — 来自 yolo_app，消除检测框抖动和闪烁
"""
import numpy as np


class SimpleTracker:
    """类别感知 IOU 跟踪器 + EMA 坐标平滑

    功能:
    1. IOU 匹配相邻帧同一目标
    2. EMA(指数移动平均)平滑检测框坐标
    3. 新目标需连续命中 MIN_HITS 帧才显示（过滤误检）
    4. 目标消失超过 MAX_MISSING 帧后移除
    """

    def __init__(self, alpha=0.55, match_iou=0.25, max_missing=4, min_hits=2):
        """
        Args:
            alpha: EMA 平滑系数 (0~1)，越小越平滑
            match_iou: IOU 匹配阈值
            max_missing: 最大丢失帧数
            min_hits: 最少命中帧数（低于此值不显示）
        """
        self.alpha = alpha
        self.match_iou = match_iou
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.tracks = []
        self._next_id = 0

    def update(self, detections):
        """
        Args:
            detections: 检测结果列表，每项为 dict:
                {"bbox": [x1,y1,x2,y2], "confidence": float,
                 "class_id": int, "class_name": str}
        Returns:
            平滑后的检测结果（同格式），过滤了闪烁和误检
        """
        if not detections:
            self._age_tracks()
            return self._get_visible()

        if not self.tracks:
            for d in detections:
                self._create(d)
            return self._get_visible()

        n_t, n_d = len(self.tracks), len(detections)
        iou_mat = np.zeros((n_t, n_d), dtype=np.float32)

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                if t['class_id'] != d['class_id']:
                    iou_mat[i, j] = 0.0
                else:
                    iou_mat[i, j] = self._iou(t['sbox'], d['bbox'])

        matched_t, matched_d = set(), set()
        pairs = []
        for i in range(n_t):
            for j in range(n_d):
                if iou_mat[i, j] >= self.match_iou:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for _, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            a = self.alpha
            t = self.tracks[ti]
            d = detections[di]
            t['sbox'] = [a * d['bbox'][k] + (1 - a) * t['sbox'][k] for k in range(4)]
            t['confidence'] = d['confidence']
            t['class_id'] = d['class_id']
            t['class_name'] = d.get('class_name', t.get('class_name', ''))
            t['missing'] = 0
            t['hits'] += 1

        for di in range(n_d):
            if di not in matched_d:
                self._create(detections[di])

        for ti in range(n_t):
            if ti not in matched_t:
                self.tracks[ti]['missing'] += 1

        self.tracks = [t for t in self.tracks if t['missing'] < self.max_missing]
        return self._get_visible()

    def _create(self, det):
        self.tracks.append({
            'id': self._next_id,
            'sbox': list(det['bbox']),
            'confidence': det['confidence'],
            'class_id': det['class_id'],
            'class_name': det.get('class_name', ''),
            'missing': 0,
            'hits': 1,
        })
        self._next_id += 1

    def _age_tracks(self):
        for t in self.tracks:
            t['missing'] += 1
        self.tracks = [t for t in self.tracks if t['missing'] < self.max_missing]

    def _get_visible(self):
        """返回连续出现 >= min_hits 帧且当前未丢失的轨迹"""
        return [
            {"bbox": [int(round(b)) for b in t['sbox']],
             "confidence": t['confidence'],
             "class_id": t['class_id'],
             "class_name": t['class_name']}
            for t in self.tracks
            if t['hits'] >= self.min_hits and t['missing'] <= 1
        ]

    @staticmethod
    def _iou(bbox1, bbox2):
        """计算两个边界框的 IOU"""
        x1, y1, x2, y2 = bbox1
        x1b, y1b, x2b, y2b = bbox2
        xi1 = max(x1, x1b)
        yi1 = max(y1, y1b)
        xi2 = min(x2, x2b)
        yi2 = min(y2, y2b)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2b - x1b) * (y2b - y1b)
        union = area1 + area2 - inter
        return inter / (union + 1e-6)
