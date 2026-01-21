from typing import Optional
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def mask_to_polygons(mask: np.ndarray, simplify_tolerance: float = 1.0) -> list[Polygon]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue

        #convert to polygon coordinates
        coords = contour.reshape(-1, 2).tolist()

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)

            if poly.area > 0:
                if simplify_tolerance > 0:
                    poly = poly.simplify(simplify_tolerance, preserve_topology=True)
                polygons.append(poly)
        except Exception:
            continue

    return polygons


def merge_polygons(polygons: list[Polygon], distance: float = 0.0) -> list[Polygon]:
    if not polygons:
        return []

    if distance <= 0:
        return polygons

    #union nearby polygons
    buffered = [p.buffer(distance) for p in polygons]
    merged = unary_union(buffered)

    if hasattr(merged, "geoms"):
        return list(merged.geoms)
    else:
        return [merged]


def vectorize_mask(mask: np.ndarray, simplify_tolerance: float = 1.0, merge_distance: float = 0.0) -> list[Polygon]:
    polygons = mask_to_polygons(mask, simplify_tolerance)

    if merge_distance > 0:
        polygons = merge_polygons(polygons, merge_distance)

    return polygons





