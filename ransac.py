import numpy as np


# ── Affine solver ─────────────────────────────────────────────────────────────

def solve_affine(src_pts, dst_pts):
    """
    Solve for a 2x3 affine matrix M via least squares (Ax = b).

    For n point pairs the system is:
        [x  y  1  0  0  0] [a ]   [u]
        [0  0  0  x  y  1] [b ] = [v]
                            [tx]
                            [c ]
                            [d ]
                            [ty]

    Returns 2x3 matrix [[a, b, tx], [c, d, ty]].
    """
    n = len(src_pts)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)

    for i, ((x, y), (u, v)) in enumerate(zip(src_pts, dst_pts)):
        A[2 * i]     = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i]     = u
        b[2 * i + 1] = v

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, tx, c, d, ty = coeffs
    return np.array([[a, b_, tx],
                     [c, d,  ty]], dtype=np.float64)


def apply_affine(M, pts):
    """Apply 2x3 affine matrix to (N, 2) array. Returns (N, 2)."""
    pts = np.asarray(pts, dtype=np.float64)
    return (M[:, :2] @ pts.T + M[:, 2:3]).T


# ── Validation helpers ────────────────────────────────────────────────────────

def is_valid_affine(M, min_det=0.005, max_det=25.0, max_sv_ratio=4.0):
    """
    Geometric sanity checks on the estimated affine matrix:

    1. Determinant bounds — |det| outside [min_det, max_det] means the
       transform is either collapsed (near-zero) or absurdly scaled.

    2. Singular value ratio — if the largest singular value is >> the smallest,
       the transform is heavily shearing the object. We reject sv[0]/sv[1] > 4.
    """
    det = abs(np.linalg.det(M[:, :2]))
    if det < min_det or det > max_det:
        return False

    sv = np.linalg.svd(M[:, :2], compute_uv=False)
    if sv[1] < 1e-9 or (sv[0] / sv[1]) > max_sv_ratio:
        return False

    return True


def is_collinear(pts, threshold=5.0):
    """True if three 2D points are approximately collinear (triangle area < threshold)."""
    p0, p1, p2 = np.asarray(pts, dtype=np.float64)[:3]
    area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
               (p2[0] - p0[0]) * (p1[1] - p0[1]))
    return area < threshold


def validate_corners(corners, scene_shape, template_area, area_ratio_range=(0.02, 4.0)):
    """
    Check that the projected bounding polygon makes geometric sense in the scene:

    1. At least 3 of the 4 corners must be within the scene boundary (with margin).
    2. Polygon area (Shoelace formula) must be within area_ratio_range relative
       to the template object's area — catches boxes that are tiny or enormous.
    3. Centroid of the polygon must be inside the scene image.
    4. Bounding-box diagonal must not exceed the scene's diagonal.
    """
    sh, sw = scene_shape[:2]
    margin = max(sw, sh) * 0.15

    # 1. Corner containment
    inside = sum(-margin <= x < sw + margin and -margin <= y < sh + margin
                 for x, y in corners)
    if inside < 3:
        return False

    xs = corners[:, 0].astype(float)
    ys = corners[:, 1].astype(float)
    n  = len(xs)

    # 2. Polygon area via Shoelace formula
    area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                         for i in range(n)))
    if template_area > 0:
        ratio = area / template_area
        if ratio < area_ratio_range[0] or ratio > area_ratio_range[1]:
            return False

    # 3. Centroid inside scene
    if not (0 <= xs.mean() < sw and 0 <= ys.mean() < sh):
        return False

    # 4. Diagonal check
    bbox_diag  = ((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2) ** 0.5
    scene_diag = (sw ** 2 + sh ** 2) ** 0.5
    if bbox_diag > scene_diag:
        return False

    return True


# ── RANSAC ────────────────────────────────────────────────────────────────────

def ransac_affine(src_pts, dst_pts, n_iter=2000, threshold=10.0):
    """
    RANSAC affine estimator.

    Each iteration:
      1. Sample 3 point pairs
      2. Reject collinear samples
      3. Solve affine with least squares
      4. Reject degenerate/sheared transforms
      5. Count inliers (reprojection error < threshold)

    Best model is refined on all its inliers.

    src_pts: (N, 2) template keypoint positions
    dst_pts: (N, 2) scene keypoint positions
    Returns (M [2x3], inlier_mask [bool array]).
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    n = len(src_pts)

    if n < 3:
        return None, np.zeros(n, dtype=bool)

    best_M    = None
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx   = rng.choice(n, 3, replace=False)
        s3, d3 = src_pts[idx], dst_pts[idx]

        if is_collinear(s3) or is_collinear(d3):
            continue

        try:
            M = solve_affine(s3, d3)
        except Exception:
            continue

        if not is_valid_affine(M):
            continue

        errors = np.linalg.norm(apply_affine(M, src_pts) - dst_pts, axis=1)
        mask   = errors < threshold
        count  = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask  = mask
            best_M     = M

    # Refinement: refit on all inliers
    if best_count >= 3:
        try:
            M_ref = solve_affine(src_pts[best_mask], dst_pts[best_mask])
            if is_valid_affine(M_ref):
                errors    = np.linalg.norm(apply_affine(M_ref, src_pts) - dst_pts, axis=1)
                best_mask = errors < threshold
                best_M    = M_ref
        except Exception:
            pass

    return best_M, best_mask
