import numpy as np


def compute_vote(t_kp, s_kp, template_center):
    """
    For a matched pair (template kp, scene kp), predict the object's center
    in the scene via the 4D Generalized Hough Transform.

    The vector from the template keypoint to the template center is:
        [dx, dy] = template_center - t_kp.pt

    After rotating by the relative orientation and scaling by the relative size,
    this same vector (in scene coordinates) points from the scene keypoint to
    the predicted object center:

        predicted_center = s_kp.pt + scale * R(angle_diff) * [dx, dy]

    Returns (pred_cx, pred_cy, scale_ratio, angle_diff).
    """
    tx, ty = t_kp.pt
    sx, sy = s_kp.pt
    cx, cy = template_center

    scale_ratio = s_kp.size / max(t_kp.size, 1e-9)
    angle_diff  = s_kp.angle - t_kp.angle

    # Vector: template keypoint  →  template center
    dx = cx - tx
    dy = cy - ty

    # Rotate that vector by angle_diff, then scale into scene coordinates
    theta  = np.deg2rad(angle_diff)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx_s = scale_ratio * (cos_t * dx - sin_t * dy)
    dy_s = scale_ratio * (sin_t * dx + cos_t * dy)

    pred_cx = sx + dx_s
    pred_cy = sy + dy_s

    return pred_cx, pred_cy, scale_ratio, angle_diff


def build_hough_space(matches, kp_template, kp_scene, template_center,
                       bin_xy=50, bin_scale=0.5, bin_angle=45):
    """
    Build a 4D Hough accumulator from the current active matches.
    Each match casts one vote at (cx_bin, cy_bin, scale_bin, angle_bin).

    Returns a dict mapping bin_key -> list of match indices.
    """
    hough = {}

    for i, (s_idx, t_idx) in enumerate(matches):
        pred_cx, pred_cy, scale_ratio, angle_diff = compute_vote(
            kp_template[t_idx], kp_scene[s_idx], template_center
        )

        cx_bin    = int(round(pred_cx / bin_xy))
        cy_bin    = int(round(pred_cy / bin_xy))
        # log2 gives symmetric bins: 0.5x and 2x are equidistant from 1x
        scale_bin = int(round(np.log2(max(scale_ratio, 1e-3)) / bin_scale))
        angle_bin = int(round((angle_diff % 360) / bin_angle))

        key = (cx_bin, cy_bin, scale_bin, angle_bin)
        hough.setdefault(key, []).append(i)

    return hough


def get_best_bin(hough_space):
    """Return the bin key with the most votes, or None if the space is empty."""
    if not hough_space:
        return None
    return max(hough_space, key=lambda k: len(hough_space[k]))
