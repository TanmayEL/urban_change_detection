"""Tests for change detection module."""

import numpy as np
import pytest

from ugcd.change import (
    compute_absolute_difference,
    detect_change,
    threshold_change,
)


def test_compute_absolute_difference():
    """test absolute difference computation"""
    img1 = np.array([[100, 150], [200, 50]], dtype=np.uint8)
    img2 = np.array([[120, 130], [180, 70]], dtype=np.uint8)

    diff = compute_absolute_difference(img1, img2)

    expected = np.array([[20, 20], [20, 20]], dtype=np.uint8)
    np.testing.assert_array_equal(diff, expected)


def test_compute_absolute_difference_shape_mismatch():
    """Test that shape mismatch raises error"""
    img1 = np.array([[100, 150]], dtype=np.uint8)
    img2 = np.array([[120, 130], [180, 70]], dtype=np.uint8)

    with pytest.raises(ValueError, match="shapes must match"):
        compute_absolute_difference(img1, img2)


def test_threshold_change():
    """Test threhsolding of difference image"""
    diff = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    mask = threshold_change(diff, threshold=25.0)

    expected = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_detect_change():
    #create synthetic images with known changes
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img2 = np.zeros((10, 10), dtype=np.uint8)

    #add a bright region in T2
    img2[3:7, 3:7] = 100

    change_mask, diff_img = detect_change(img1, img2, threshold=50.0)

    assert np.sum(change_mask > 0) > 0
    assert np.sum(diff_img > 0) > 0
    assert change_mask.shape == img1.shape
    assert diff_img.shape == img1.shape







