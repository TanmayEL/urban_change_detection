import numpy as np


def compute_absolute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    #get |T1 - T2| for each pixel
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    return np.abs(img1.astype(np.float32) - img2.astype(np.float32)).astype(np.uint8)


def threshold_change(diff_img: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    mask = np.zeros_like(diff_img, dtype=np.uint8)
    mask[diff_img > threshold] = 255
    return mask


def detect_change(img1: np.ndarray, img2: np.ndarray, threshold: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    #compute diff and threshold
    diff_img = compute_absolute_difference(img1, img2)
    change_mask = threshold_change(diff_img, threshold)
    return change_mask, diff_img





