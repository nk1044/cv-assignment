import numpy as np


def match_features(desc_scene, desc_template, ratio=0.75):
    """
    Match scene descriptors to template descriptors using Lowe's ratio test.
    NO cv2 matchers used — distances are computed from scratch via NumPy.

    Match direction: scene -> template
    This is the key trick for multi-instance detection: every scene keypoint can
    independently vote for the object's location, so all instances get votes even
    though the template only appears once.

    Algorithm:
      For each scene descriptor, find its 2 nearest template descriptors.
      Accept the match only when: dist_to_best / dist_to_second < ratio
      This rejects ambiguous matches that look similar to background noise.

    Returns list of (scene_idx, template_idx) pairs.
    """
    if len(desc_scene) == 0 or len(desc_template) == 0:
        return []

    # Efficient pairwise Euclidean distance using the identity:
    #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a · b)
    a2 = np.sum(desc_scene ** 2, axis=1, keepdims=True)   # (N, 1)
    b2 = np.sum(desc_template ** 2, axis=1, keepdims=True) # (M, 1)
    ab = desc_scene @ desc_template.T                       # (N, M)
    dist_sq = np.maximum(a2 + b2.T - 2 * ab, 0.0)         # clip negatives from float error
    distances = np.sqrt(dist_sq)                            # (N, M)

    matches = []
    for i in range(len(desc_scene)):
        # Partial sort: get indices of the 2 smallest distances
        if distances.shape[1] < 2:
            break
        top2_idx = np.argpartition(distances[i], 2)[:2]
        top2_idx = top2_idx[np.argsort(distances[i, top2_idx])]  # sort those 2
        best, second = top2_idx[0], top2_idx[1]

        d1 = distances[i, best]
        d2 = distances[i, second]

        # Lowe's ratio test
        if d2 > 1e-6 and (d1 / d2) < ratio:
            matches.append((int(i), int(best)))

    return matches
