from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import rasterio


def load_image(path: Union[str, Path], as_grayscale: bool = True) -> Tuple[np.ndarray, dict]:
    #returns image array and metadata dict
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    metadata = {}

    #try GeoTIFF first
    if path.suffix.lower() in [".tif", ".tiff"]:
        try:
            with rasterio.open(path) as src:
                img = src.read()
                #Handle multiband: if RGB, transpose to (H, W, C)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                elif img.shape[0] == 1:
                    img = img[0]
                else:
                    img = img[0]

                if img.dtype != np.uint8:
                    img = (img / img.max() * 255).astype(np.uint8)

                if as_grayscale and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                metadata["crs"] = src.crs
                metadata["transform"] = src.transform
                metadata["bounds"] = src.bounds
                metadata["width"] = src.width
                metadata["height"] = src.height

                return img, metadata
        except Exception as e:
            print(f"Warning: Could not read as GeoTIFF ({e}), trying OpenCV...")

    if as_grayscale:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    return img, metadata


def save_image(path: Union[str, Path], img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    #convert RGB to BGR for openCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), img)


def load_rgb_bands(
    red_path: Union[str, Path],
    green_path: Union[str, Path],
    blue_path: Union[str, Path],
    as_grayscale: bool = True,
) -> Tuple[np.ndarray, dict]:
    #load RGB bands from separate files n combines them into single image and normalizes uint16 to uint8
    red_path = Path(red_path)
    green_path = Path(green_path)
    blue_path = Path(blue_path)

    for path, name in [
        (red_path, "Red"),
        (green_path, "Green"),
        (blue_path, "Blue"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} band not found: {path}")

    metadata = {}

    bands = []
    for i, (path, name) in enumerate([(red_path, "Red"), (green_path, "Green"), (blue_path, "Blue")]):
        try:
            with rasterio.open(path) as src:
                band = src.read(1)

                #save geo metadata from red band
                if i == 0:
                    metadata["crs"] = src.crs
                    metadata["transform"] = src.transform
                    metadata["bounds"] = src.bounds
                    metadata["width"] = src.width
                    metadata["height"] = src.height

                if i > 0 and band.shape != bands[0].shape:
                    raise ValueError(f"Band dimensions mismatch: {name} band shape {band.shape} does not match Red band shape {bands[0].shape}")

                if band.dtype != np.uint8:
                    if band.dtype == np.uint16:
                        band = (band.astype(np.float32) / 10000.0 * 255).clip(0, 255).astype(np.uint8)
                    else:
                        band_max = band.max()
                        if band_max > 0:
                            band = (band.astype(np.float32) / band_max * 255).clip(0, 255).astype(np.uint8)
                        else:
                            band = band.astype(np.uint8)

                bands.append(band)

        except Exception as e:
            raise ValueError(f"Could not load {name} band from {path}: {e}")

    #combine back
    rgb_img = np.stack(bands, axis=-1)

    if as_grayscale:
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    return rgb_img, metadata

