import cv2
import numpy as np


def extract_sift(image_path: str):
    """
    Load image and extract SIFT keypoints + 128-dim descriptors.
    Returns (bgr_image, keypoints, descriptors).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=3000)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        descriptors = np.zeros((0, 128), dtype=np.float32)

    return img, keypoints, descriptors


def get_template_object_bounds(keypoints, image_shape, pad=2.0):
    """
    Compute the tight bounding box of the template object using keypoint positions.

    Instead of using the full image corners (which include background/table/wall),
    we use the spread of SIFT keypoints — they cluster on the actual object.
    Percentile filtering (2nd–98th) removes any stray background keypoints.

    Returns (corners [4x2 float64], center (cx, cy)).
    """
    h, w = image_shape[:2]
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float64)

    if len(pts) >= 10:
        x1, x2 = np.percentile(pts[:, 0], [2, 98])
        y1, y2 = np.percentile(pts[:, 1], [2, 98])
        # Keep points within the percentile range (with small slack)
        mask = (pts[:, 0] >= x1 - 20) & (pts[:, 0] <= x2 + 20) & \
               (pts[:, 1] >= y1 - 20) & (pts[:, 1] <= y2 + 20)
        obj_pts = pts[mask] if mask.sum() >= 5 else pts
    else:
        obj_pts = pts

    min_x = max(0.0,       float(obj_pts[:, 0].min()) - pad)
    min_y = max(0.0,       float(obj_pts[:, 1].min()) - pad)
    max_x = min(w - 1.0,  float(obj_pts[:, 0].max()) + pad)
    max_y = min(h - 1.0,  float(obj_pts[:, 1].max()) + pad)

    corners = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ], dtype=np.float64)

    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    return corners, center


def match_features(desc_scene, desc_template, ratio=0.75):
    """
    Match scene descriptors to template descriptors using Lowe's ratio test.
    No cv2 matchers — all distances computed from scratch via NumPy.

    Direction: scene -> template, so every scene keypoint independently votes
    for an object location (enables multi-instance detection).

    Returns list of (scene_idx, template_idx) pairs.
    """
    if len(desc_scene) == 0 or len(desc_template) == 0:
        return []

    # Vectorised pairwise L2 via the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a·b)
    a2 = np.sum(desc_scene ** 2, axis=1, keepdims=True)    # (N, 1)
    b2 = np.sum(desc_template ** 2, axis=1, keepdims=True)  # (M, 1)
    ab = desc_scene @ desc_template.T                        # (N, M)
    distances = np.sqrt(np.maximum(a2 + b2.T - 2 * ab, 0.0))  # (N, M)

    if distances.shape[1] < 2:
        return []

    # For each scene descriptor, get the 2 closest template descriptors
    idx2 = np.argpartition(distances, 2, axis=1)[:, :2]
    rows = np.arange(len(distances))
    d1 = distances[rows, idx2[:, 0]]
    d2 = distances[rows, idx2[:, 1]]

    # Make sure d1 <= d2 (argpartition does not sort the top-2)
    swap = d1 > d2
    idx2[swap, 0], idx2[swap, 1] = idx2[swap, 1].copy(), idx2[swap, 0].copy()
    d1[swap], d2[swap] = d2[swap].copy(), d1[swap].copy()

    matches = []
    for i in range(len(distances)):
        if d2[i] > 1e-9 and (d1[i] / d2[i]) < ratio:
            matches.append((int(i), int(idx2[i, 0])))

    return matches
