"""SHADE RAW-domain processing pipeline.

Wraps the full pre/post-processing steps around :class:`SHADENet` so that
callers can pass a raw file path and receive an enhanced sRGB image without
worrying about pack/unpack logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from .model import SHADENet
from .utils import pack_bayer, postprocess


class SHADEPipeline:
    """End-to-end inference pipeline for SHADE.

    Args:
        weights: Path to a ``.pth`` checkpoint produced by ``scripts/train.py``.
        device: PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
            Defaults to CUDA if available, otherwise CPU.
    """

    def __init__(
        self,
        weights: Union[str, Path, None] = None,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = SHADENet()
        if weights is not None:
            state = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def enhance(self, raw_path: Union[str, Path]) -> np.ndarray:
        """Enhance a single RAW file.

        Args:
            raw_path: Path to the input ``.dng`` / ``.raw`` file.

        Returns:
            Enhanced sRGB image as a ``uint8`` NumPy array of shape
            ``(H, W, 3)``.
        """
        import rawpy  # imported lazily so the package stays optional for tests

        with rawpy.imread(str(raw_path)) as raw:
            bayer = raw.raw_image_visible.astype(np.float32)
            black = raw.black_level_per_channel
            white = raw.white_level

        # Normalise
        bayer = (bayer - np.mean(black)) / (white - np.mean(black))
        bayer = np.clip(bayer, 0.0, 1.0)

        packed = pack_bayer(bayer)                                 # (4, H/2, W/2)
        tensor = torch.from_numpy(packed).unsqueeze(0).to(self.device)  # (1,4,H/2,W/2)

        output = self.model(tensor)                                # (1, 3, H/2, W/2)
        return postprocess(output.squeeze(0).cpu().numpy())
