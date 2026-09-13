import numpy as np


def _aabb(corners):
    """Axis-aligned bounding box of a polygon as (x1, y1, x2, y2)."""
    corners = np.asarray(corners, dtype=float)
    return (corners[:, 0].min(), corners[:, 1].min(),
            corners[:, 0].max(), corners[:, 1].max())


def iou(corners1, corners2):
    """
    Intersection over Union between two bounding polygons.
    Approximated using their axis-aligned bounding boxes.
    """
    x1a, y1a, x2a, y2a = _aabb(corners1)
    x1b, y1b, x2b, y2b = _aabb(corners2)

    iw = max(0.0, min(x2a, x2b) - max(x1a, x1b))
    ih = max(0.0, min(y2a, y2b) - max(y1a, y1b))
    inter = iw * ih

    if inter == 0:
        return 0.0

    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union  = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def non_max_suppression(detections, iou_threshold=0.3):
    """
    Remove duplicate detections of the same object instance.
    Higher-inlier-count detections take priority.
    Any lower-confidence detection overlapping a kept one by > iou_threshold is dropped.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d['inliers'], reverse=True)
    kept = []

    for det in detections:
        if not any(iou(det['corners'], k['corners']) > iou_threshold for k in kept):
            kept.append(det)

    return kept
