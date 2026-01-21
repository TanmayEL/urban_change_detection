from typing import Optional

import numpy as np
from shapely.geometry import Polygon


def compute_change_statistics(change_mask: np.ndarray, polygons: Optional[list[Polygon]] = None, pixel_size_m2: Optional[float] = None) -> dict:
    #calc changed pixels, percentage, polygon count, area
    total_pixels = change_mask.size
    changed_pixels = np.sum(change_mask > 0)
    changed_percent = (changed_pixels / total_pixels) * 100.0

    stats = {
        "changed_pixels": int(changed_pixels),
        "total_pixels": int(total_pixels),
        "changed_percent": float(changed_percent),
        "num_polygons": len(polygons) if polygons else 0,
    }

    if pixel_size_m2 is not None:
        changed_area_m2 = changed_pixels * pixel_size_m2
        stats["changed_area_m2"] = float(changed_area_m2)

    if polygons:
        total_polygon_area_pixels = sum(p.area for p in polygons)
        stats["total_polygon_area_pixels"] = float(total_polygon_area_pixels)
        if pixel_size_m2 is not None:
            stats["total_polygon_area_m2"] = (
                total_polygon_area_pixels * pixel_size_m2
            )

    return stats





