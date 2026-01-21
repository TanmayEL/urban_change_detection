from pathlib import Path
from typing import Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    #change mask on image (red by default)
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()

    colored_mask = np.zeros_like(img_rgb)
    colored_mask[mask > 0] = color

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, colored_mask, alpha, 0)

    return overlay


def save_comparison_figure(img1: np.ndarray, img2: np.ndarray, change_mask: np.ndarray, overlay: np.ndarray, output_path: Union[str, Path], figsize: tuple[int, int] = (16, 4)) -> None:
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    def to_rgb(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    axes[0].imshow(to_rgb(img1))
    axes[0].set_title("Time T1")
    axes[0].axis("off")

    axes[1].imshow(to_rgb(img2))
    axes[1].set_title("Time T2")
    axes[1].axis("off")

    axes[2].imshow(change_mask, cmap="gray")
    axes[2].set_title("Change Mask")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()





