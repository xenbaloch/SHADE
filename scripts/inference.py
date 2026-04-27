"""Inference script for the SHADE pipeline.

Example
-------
python scripts/inference.py \\
    --input  samples/backlit.dng \\
    --output results/enhanced.png \\
    --weights checkpoints/shade_weights.pth
"""

import argparse
from pathlib import Path

from PIL import Image

from shade import SHADEPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHADE: enhance a backlit / low-light RAW image."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input RAW file (.dng, etc.)")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG file path")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to trained SHADE weights (.pth). "
        "If omitted, the network runs with random weights (for testing only).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device string, e.g. 'cuda' or 'cpu'. "
        "Auto-detected when not specified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = SHADEPipeline(weights=args.weights, device=args.device)

    print(f"Processing: {args.input}")
    enhanced = pipeline.enhance(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced).save(args.output)
    print(f"Saved enhanced image to: {args.output}")


if __name__ == "__main__":
    main()
