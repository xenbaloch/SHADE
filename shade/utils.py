"""Utility helpers for the SHADE pipeline."""

from __future__ import annotations

import numpy as np


def pack_bayer(bayer: np.ndarray) -> np.ndarray:
    """Pack a full-resolution Bayer array into 4-channel half-resolution form.

    Assumes RGGB Bayer pattern.  The output shape is ``(4, H//2, W//2)``.

    Args:
        bayer: Float32 Bayer image of shape ``(H, W)``.

    Returns:
        Packed array of shape ``(4, H//2, W//2)``.
    """
    h, w = bayer.shape
    packed = np.stack(
        [
            bayer[0:h:2, 0:w:2],   # R
            bayer[0:h:2, 1:w:2],   # Gr
            bayer[1:h:2, 0:w:2],   # Gb
            bayer[1:h:2, 1:w:2],   # B
        ],
        axis=0,
    )
    return packed.astype(np.float32)


def postprocess(tensor: np.ndarray) -> np.ndarray:
    """Convert a CHW float32 tensor in ``[0, 1]`` to a HWC uint8 image.

    Args:
        tensor: Array of shape ``(3, H, W)`` with values in ``[0, 1]``.

    Returns:
        ``uint8`` array of shape ``(H, W, 3)``.
    """
    img = np.clip(tensor, 0.0, 1.0)
    img = (img * 255).round().astype(np.uint8)
    return img.transpose(1, 2, 0)  # CHW -> HWC


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        pred:   Predicted image, ``float32`` or ``uint8``.
        target: Ground-truth image, same dtype and shape as *pred*.

    Returns:
        PSNR value in decibels.  Returns ``float('inf')`` for identical inputs.
    """
    pred_f = pred.astype(np.float64)
    target_f = target.astype(np.float64)

    # Normalize uint8 images to [0, 1]
    if pred.dtype == np.uint8:
        pred_f /= 255.0
        target_f /= 255.0

    mse = np.mean((pred_f - target_f) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))
