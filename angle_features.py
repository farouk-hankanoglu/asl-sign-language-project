

import numpy as np

FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}

FINGER_BASES = {
    "thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17
}
ADJACENT_FINGER_PAIRS = [
    ("thumb", "index"), ("index", "middle"),
    ("middle", "ring"), ("ring", "pinky"),
]


def _angle_between(v1, v2):
    """Angle in radians between two vectors, robust to zero-length vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def landmarks_to_angles(flat_63):
    """
    flat_63: array-like of 63 values (21 points x, y, z), as produced
    by extract_landmarks.py.
    Returns: a list of 19 rotation-invariant angle features (radians):
        - 3 bend angles per finger x 5 fingers = 15
        - 4 spread angles between adjacent fingers = 4
    """
    pts = np.array(flat_63, dtype=np.float32).reshape(21, 3)

    features = []

    for finger, chain in FINGER_CHAINS.items():
        for i in range(1, len(chain) - 1):
            p_prev = pts[chain[i - 1]]
            p_curr = pts[chain[i]]
            p_next = pts[chain[i + 1]]
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            features.append(_angle_between(v1, v2))

    wrist = pts[0]
    for f1, f2 in ADJACENT_FINGER_PAIRS:
        v1 = pts[FINGER_BASES[f1]] - wrist
        v2 = pts[FINGER_BASES[f2]] - wrist
        features.append(_angle_between(v1, v2))

    return features


def angle_feature_names():
    names = []
    for finger, chain in FINGER_CHAINS.items():
        for i in range(1, len(chain) - 1):
            names.append(f"{finger}_joint{i}_bend")
    for f1, f2 in ADJACENT_FINGER_PAIRS:
        names.append(f"{f1}_{f2}_spread")
    return names


if __name__ == "__main__":
    # quick self-test with a dummy flat hand shape
    dummy = np.random.uniform(-1, 1, 63)
    angles = landmarks_to_angles(dummy)
    names = angle_feature_names()
    print(f"Produced {len(angles)} angle features (expected 19)")
    for n, a in zip(names, angles):
        print(f"  {n}: {a:.3f} rad")
