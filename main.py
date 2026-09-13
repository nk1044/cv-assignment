"""
Assignment 1: Multi-Instance Object Recognition in Cluttered Scenes
Run with:  uv run python main.py
Outputs:   outputs/
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

from pipeline import run_pipeline

TEMPLATE_PATH = "images/template.jpg"
SCENE_PATH    = "images/scene.jpg"
OUT_DIR       = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

COLOURS = [
    (0,   220,   0),    # green
    (0,   140, 255),    # orange
    (220,   0, 220),    # magenta
    (0,   220, 220),    # cyan
]


def clip_to_image(corners, shape):
    """Clip corner coordinates to image bounds for clean polygon drawing."""
    h, w = shape[:2]
    c = corners.astype(np.float32).copy()
    c[:, 0] = np.clip(c[:, 0], 0, w - 1)
    c[:, 1] = np.clip(c[:, 1], 0, h - 1)
    return c.astype(np.int32)


def save_naive_matches(template_img, scene_img, matches, kp_t, kp_s,
                        path, max_lines=150):
    """Side-by-side view of raw matches — shows the chaotic baseline."""
    h1, w1 = template_img.shape[:2]
    h2, w2 = scene_img.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1]      = template_img
    canvas[:h2, w1:w1+w2] = scene_img

    rng = np.random.default_rng(0)
    for s_idx, t_idx in matches[:max_lines]:
        t_pt = (int(kp_t[t_idx].pt[0]),       int(kp_t[t_idx].pt[1]))
        s_pt = (int(kp_s[s_idx].pt[0]) + w1,  int(kp_s[s_idx].pt[1]))
        color = tuple(int(c) for c in rng.integers(60, 220, 3))
        cv2.line(canvas, t_pt, s_pt, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, t_pt, 3, (0, 255, 255), -1)
        cv2.circle(canvas, s_pt, 3, (0, 255, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Template  ({len(matches)} matches after Lowe's test)",
                (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Scene (no geometric verification)",
                (w1 + 10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(path), canvas)
    print(f"Saved: {path}")


def save_final_detections(scene_img, detections, path):
    """Draw compact, fitted bounding boxes on each detected instance."""
    result = scene_img.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX

    if not detections:
        cv2.putText(result, "No instances detected", (30, 50),
                    font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        for i, det in enumerate(detections):
            colour  = COLOURS[i % len(COLOURS)]
            corners = clip_to_image(det['corners'], result.shape)

            cv2.polylines(result, [corners.reshape(-1, 1, 2)],
                          True, colour, 3, cv2.LINE_AA)

            # Label: drop-shadow for readability on any background
            xs, ys = corners[:, 0], corners[:, 1]
            lx = int(np.clip(xs.min(), 0, result.shape[1] - 1))
            ly = int(np.clip(ys.min() - 8, 10, result.shape[0] - 1))
            label = f"#{i+1}  inliers={det['inliers']}"
            cv2.putText(result, label, (lx + 1, ly + 1),
                        font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(result, label, (lx, ly),
                        font, 0.65, colour, 2, cv2.LINE_AA)

    cv2.imwrite(str(path), result)
    print(f"Saved: {path}")


def main():
    print("=" * 55)
    print("  Multi-Instance Object Recognition Pipeline")
    print("=" * 55 + "\n")

    scene_img, template_img, detections, matches, kp_s, kp_t = run_pipeline(
        TEMPLATE_PATH,
        SCENE_PATH,
        lowe_ratio=0.75,
        hough_bin_xy=50,
        hough_bin_scale=0.5,
        hough_bin_angle=45,
        hough_min_votes=3,
        ransac_iter=2000,
        ransac_thresh=10.0,
        min_inliers=4,
        iou_threshold=0.3,
    )

    print(f"\nTotal instances detected: {len(detections)}")
    for i, d in enumerate(detections):
        print(f"  Instance {i+1}: {d['inliers']} inliers, "
              f"mean reproj error = {d['mean_error']:.2f} px")

    save_naive_matches(template_img, scene_img, matches, kp_t, kp_s,
                       OUT_DIR / "naive_matches.jpg")
    save_final_detections(scene_img, detections,
                          OUT_DIR / "final_detections.jpg")

    print("\nDone. Check the outputs/ folder.")


if __name__ == "__main__":
    main()
