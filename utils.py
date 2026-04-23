import numpy as np


def normalize_landmarks(landmarks):
    """
    Normalizes MediaPipe hand landmarks for ML classification.

    Steps:
      1. Validate that exactly 21 landmarks are present (C.1.3).
      2. Translate: subtract the wrist (index 0) from all points.
      3. Scale: divide by the maximum wrist-to-landmark distance.
         If max_dist == 0 (degenerate pose), return a zero vector (C.1.4).
      4. Flatten to a 63-element feature vector.

    Args:
        landmarks: list of 21 dicts, each with keys 'x', 'y', 'z' (floats).

    Returns:
        np.ndarray of shape (63,).

    Raises:
        ValueError: if the landmark count is not exactly 21.
    """
    # C.1.3: Hard guard — never let an incorrect feature length reach the model
    if len(landmarks) != 21:
        raise ValueError(
            f"normalize_landmarks expects exactly 21 landmarks, got {len(landmarks)}."
        )

    coords = np.array([[l['x'], l['y'], l['z']] for l in landmarks], dtype=np.float32)

    # 1. Translate: move wrist to origin
    wrist = coords[0].copy()
    translated = coords - wrist

    # 2. Scale: normalise by maximum distance from wrist
    max_dist = np.max(np.linalg.norm(translated, axis=1))

    # C.1.4: Degenerate pose (all landmarks collapsed to a point) → zero vector
    if max_dist == 0:
        return np.zeros(63, dtype=np.float32)

    scaled = translated / max_dist

    # 3. Flatten to (63,)
    return scaled.flatten()
