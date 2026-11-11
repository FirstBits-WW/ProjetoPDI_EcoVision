import numpy as np
import pytest

from app.services.image_processing import (
    InvalidImageError,
    create_variants,
    decode_image,
    preprocess_image,
)


def test_create_variants_keeps_image_size_and_brightens_with_gamma() -> None:
    image = np.full((24, 32, 3), 64, dtype=np.uint8)

    variants = create_variants(image, gamma=1.25)

    assert set(variants) == {"original", "clahe", "gamma"}
    assert all(variant.shape == image.shape for variant in variants.values())
    assert variants["gamma"][0, 0, 0] > image[0, 0, 0]


def test_preprocess_image_applies_brightness_and_optional_denoising() -> None:
    image = np.full((24, 32, 3), 64, dtype=np.uint8)

    result = preprocess_image(
        image,
        brightness=20,
        clahe_enabled=True,
        denoise=True,
    )

    assert result.shape == image.shape
    assert result[12, 16, 0] > image[12, 16, 0]


def test_decode_image_rejects_non_image_bytes() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image")
