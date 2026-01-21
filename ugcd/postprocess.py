from typing import Optional
import cv2
import numpy as np
from skimage import morphology


def remove_small_blobs(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    #filter out small blobs and find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    filtered_mask = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == label_id] = 255

    return filtered_mask


def apply_morphology(mask: np.ndarray, operation: str = "closing", kernel_size: int = 3) -> np.ndarray:
    #closing fills holes, opening removes noise
    mask_bool = mask > 0

    kernel = morphology.disk(kernel_size)

    if operation == "opening":
        result = morphology.binary_opening(mask_bool, kernel)
    elif operation == "closing":
        result = morphology.binary_closing(mask_bool, kernel)
    elif operation == "dilation":
        result = morphology.binary_dilation(mask_bool, kernel)
    elif operation == "erosion":
        result = morphology.binary_erosion(mask_bool, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}. Choose from: opening, closing, dilation, erosion"
        )

    return (result.astype(np.uint8) * 255)


def postprocess_mask(mask: np.ndarray, min_area: int = 100, morph_operation: Optional[str] = "closing", morph_kernel_size: int = 3) -> np.ndarray:
    if morph_operation:
        mask = apply_morphology(mask, morph_operation, morph_kernel_size)

    if min_area > 0:
        mask = remove_small_blobs(mask, min_area)

    return mask





