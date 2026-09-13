import numpy as np

from features import extract_sift, get_template_object_bounds, match_features
from hough import build_hough_space, get_best_bin
from ransac import ransac_affine, apply_affine, is_valid_affine, validate_corners
from nms import non_max_suppression


def run_pipeline(
    template_path,
    scene_path,
    lowe_ratio=0.75,
    hough_bin_xy=50,
    hough_bin_scale=0.5,
    hough_bin_angle=45,
    hough_min_votes=3,
    ransac_iter=2000,
    ransac_thresh=10.0,
    min_inliers=4,
    iou_threshold=0.3,
):
    """
    Full multi-instance object detection pipeline.

    Steps:
      1. SIFT feature extraction
      2. Feature matching with Lowe's ratio test (scene -> template)
      3. Greedy Extraction Loop:
           a. Build 4D Generalized Hough Transform on active matches
           b. Take the highest-vote bin
           c. RANSAC + Least-Squares affine estimation
           d. Validate affine (inliers, reprojection error, det, SVD shear)
           e. Validate projected corners (area, centroid, diagonal)
           f. Accept or discard; either way remove the bin's matches from active pool
      4. IoU Non-Maximum Suppression

    Returns: (scene_img, template_img, detections, raw_matches, kp_scene, kp_template)
    """
    # ------------------------------------------------------------------
    # 1. SIFT
    # ------------------------------------------------------------------
    template_img, kp_t, desc_t = extract_sift(template_path)
    scene_img,    kp_s, desc_s = extract_sift(scene_path)
    print(f"Template: {len(kp_t)} keypoints  |  Scene: {len(kp_s)} keypoints")

    # Tight bounding box of the template object (from keypoints, not image edges)
    tmpl_corners, tmpl_center = get_template_object_bounds(kp_t, template_img.shape)
    tmpl_area = abs((tmpl_corners[1, 0] - tmpl_corners[0, 0]) *
                    (tmpl_corners[2, 1] - tmpl_corners[0, 1]))
    print(f"Template object bounds: {tmpl_corners[0]} -> {tmpl_corners[2]}  "
          f"area={tmpl_area:.0f} px²")

    # ------------------------------------------------------------------
    # 2. Feature matching
    # ------------------------------------------------------------------
    matches = match_features(desc_s, desc_t, ratio=lowe_ratio)
    print(f"Matches after Lowe's ratio test: {len(matches)}")

    if not matches:
        print("No matches found.")
        return scene_img, template_img, [], [], kp_s, kp_t

    raw_matches = list(matches)  # keep for naive-match visualization

    # ------------------------------------------------------------------
    # 3. Greedy Extraction Loop
    # ------------------------------------------------------------------
    active_matches = list(matches)
    detections = []
    iteration  = 0

    while len(active_matches) >= hough_min_votes:
        iteration += 1
        print(f"\n[iter {iteration}] active matches: {len(active_matches)}")

        # 3a. Build Hough accumulator on current active matches
        hough = build_hough_space(
            active_matches, kp_t, kp_s, tmpl_center,
            bin_xy=hough_bin_xy, bin_scale=hough_bin_scale, bin_angle=hough_bin_angle,
        )

        # 3b. Best bin
        best_key = get_best_bin(hough)
        if best_key is None:
            break

        bin_indices = hough[best_key]
        if len(bin_indices) < hough_min_votes:
            print("  Best bin below vote threshold — stopping.")
            break

        bin_matches = [active_matches[i] for i in bin_indices]
        print(f"  Best Hough bin: {len(bin_matches)} votes")

        # 3c. RANSAC affine estimation
        src_pts = np.array([kp_t[m[1]].pt for m in bin_matches], dtype=np.float64)
        dst_pts = np.array([kp_s[m[0]].pt for m in bin_matches], dtype=np.float64)

        M, mask = ransac_affine(src_pts, dst_pts,
                                n_iter=ransac_iter, threshold=ransac_thresh)
        n_inliers = int(mask.sum())
        print(f"  RANSAC inliers: {n_inliers}")

        # 3d. Validate the affine transform
        valid = (
            M is not None
            and n_inliers >= min_inliers
            and is_valid_affine(M)
        )

        if valid:
            # Reprojection error on inliers
            proj     = apply_affine(M, src_pts[mask])
            mean_err = float(np.mean(np.linalg.norm(proj - dst_pts[mask], axis=1)))
            valid    = mean_err <= ransac_thresh * 1.5
            if not valid:
                print(f"  Rejected: mean reprojection error {mean_err:.1f} px")

        if valid:
            # 3e. Project template corners into scene
            corners_h = np.hstack([tmpl_corners,
                                   np.ones((len(tmpl_corners), 1))])
            corners_scene = (M @ corners_h.T).T  # (4, 2)

            # 3f. Geometric sanity on the projected box
            if not validate_corners(corners_scene, scene_img.shape, tmpl_area):
                print("  Rejected: projected corners failed geometry check.")
                valid = False

        if valid:
            print(f"  Accepted: {n_inliers} inliers, err={mean_err:.2f} px")
            detections.append({
                'corners': corners_scene.astype(np.int32),
                'inliers': n_inliers,
                'mean_error': mean_err,
            })

            # 3g. Sequential inlier subtraction — remove ALL scene matches whose
            # reprojection error under this affine is below threshold
            surviving = []
            for m in active_matches:
                t_h = np.array([*kp_t[m[1]].pt, 1.0])
                err = np.linalg.norm(M @ t_h - np.array(kp_s[m[0]].pt))
                if err > ransac_thresh:
                    surviving.append(m)
            removed = len(active_matches) - len(surviving)
            print(f"  Removed {removed} inlier matches; {len(surviving)} remaining.")
            active_matches = surviving

        else:
            # Discard the failing bin's matches to avoid an infinite loop
            bad = set(bin_indices)
            active_matches = [m for i, m in enumerate(active_matches) if i not in bad]

    # ------------------------------------------------------------------
    # 4. NMS
    # ------------------------------------------------------------------
    print(f"\nDetections before NMS: {len(detections)}")
    detections = non_max_suppression(detections, iou_threshold=iou_threshold)
    print(f"Final detections after NMS: {len(detections)}")

    return scene_img, template_img, detections, raw_matches, kp_s, kp_t
