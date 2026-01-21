import numpy as np
import pytest

from ugcd.postprocess import (
    apply_morphology,
    postprocess_mask,
    remove_small_blobs,
)


def test_remove_small_blobs():
    #create mask with small and large blobs
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:7, 2:7] = 255
    mask[10:20, 10:20] = 255

    filtered = remove_small_blobs(mask, min_area=50)

    assert np.sum(filtered[2:7, 2:7]) == 0

    assert np.sum(filtered[10:20, 10:20]) > 0


def test_apply_morphology():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    mask[4:6, 4:6] = 0  #small hole

    closed = apply_morphology(mask, operation="closing", kernel_size=3)

    assert np.sum(closed[4:6, 4:6]) > 0


def test_apply_morphology_invalid_operation():
    mask = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown operation"):
        apply_morphology(mask, operation="invalid")


def test_postprocess_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255  #large region
    mask[0:2, 0:2] = 255  #smalll noise blob

    processed = postprocess_mask(
        mask, min_area=50, morph_operation="closing", morph_kernel_size=3
    )

    assert np.sum(processed[0:2, 0:2]) == 0
    assert np.sum(processed[5:15, 5:15]) > 0


def test_postprocess_mask_no_morphology():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask[0:2, 0:2] = 255

    processed = postprocess_mask(
        mask, min_area=50, morph_operation=None, morph_kernel_size=3
    )
    assert np.sum(processed[0:2, 0:2]) == 0







