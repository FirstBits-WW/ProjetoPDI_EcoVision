from __future__ import annotations

import cv2
import numpy as np


class InvalidImageError(ValueError):
    pass


def decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("O arquivo enviado não contém uma imagem válida.")
    return image


def preprocess_image(
    image: np.ndarray,
    gamma: float = 1.0,
    contrast: float = 1.0,
    brightness: float = 0.0,
    clahe_enabled: bool = False,
    denoise: bool = False,
) -> np.ndarray:
    processed = cv2.GaussianBlur(image, (3, 3), 0.8) if denoise else image.copy()
    lookup_table = np.array(
        [((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    processed = cv2.LUT(processed, lookup_table)
    processed = np.clip(
        processed.astype(np.float32) * contrast
        + 128.0 * (1.0 - contrast)
        + brightness,
        0,
        255,
    ).astype(np.uint8)

    if clahe_enabled:
        lab_image = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_lightness = clahe.apply(lightness)
        processed = cv2.cvtColor(
            cv2.merge((enhanced_lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )

    return processed


def create_variants(
    image: np.ndarray, gamma: float = 1.25
) -> dict[str, np.ndarray]:
    return {
        "original": image,
        "clahe": preprocess_image(image, clahe_enabled=True),
        "gamma": preprocess_image(image, gamma=gamma),
    }


def encode_jpeg(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not success:
        raise RuntimeError("Não foi possível codificar a imagem processada.")
    import base64

    return f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}"
