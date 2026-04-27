"""Training script for SHADE.

Example
-------
python scripts/train.py \\
    --data_root /path/to/dataset \\
    --epochs 200 \\
    --batch_size 8 \\
    --lr 1e-4 \\
    --save_dir checkpoints
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from shade import SHADENet


# ---------------------------------------------------------------------------
# Dataset placeholder
# ---------------------------------------------------------------------------

class BacklitRAWDataset(Dataset):
    """Paired RAW / sRGB dataset for SHADE training.

    Expects the following directory layout::

        data_root/
            train/
                raw/    *.npy       packed Bayer tensors (float32, shape 4×H×W)
                gt/     *.npy       sRGB ground-truth   (float32, shape 3×H×W)

    Args:
        data_root: Root directory containing a ``train/`` sub-folder.
        split:     Dataset split, currently only ``"train"`` is used.
    """

    def __init__(self, data_root: Path, split: str = "train") -> None:
        import numpy as np  # noqa: F401 — checked at dataset init time

        self.raw_dir = data_root / split / "raw"
        self.gt_dir = data_root / split / "gt"
        self.samples = sorted(self.raw_dir.glob("*.npy"))
        if not self.samples:
            raise FileNotFoundError(
                f"No RAW samples found in {self.raw_dir}. "
                "Please download the dataset and place it under data_root/train/raw/."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import numpy as np

        raw_path = self.samples[idx]
        gt_path = self.gt_dir / raw_path.name

        raw = torch.from_numpy(np.load(raw_path))   # (4, H, W)
        gt = torch.from_numpy(np.load(gt_path))     # (3, H, W)
        return raw, gt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SHADE on paired RAW / sRGB data.")
    parser.add_argument("--data_root", required=True, type=Path, help="Root of the dataset")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device string, e.g. 'cuda' or 'cpu'. Auto-detected when not specified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # ---- Dataset & DataLoader ----
    dataset = BacklitRAWDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # ---- Model, optimiser, loss ----
    model = SHADENet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    criterion = nn.L1Loss()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for raw, gt in loader:
            raw, gt = raw.to(device), gt.to(device)
            optimiser.zero_grad()
            pred = model(raw)
            loss = criterion(pred, gt)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * raw.size(0)

        epoch_loss /= len(dataset)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch [{epoch:>4}/{args.epochs}]  loss={epoch_loss:.6f}  time={elapsed:.1f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = args.save_dir / "shade_best.pth"
            torch.save(model.state_dict(), ckpt_path)

    # Save final checkpoint
    torch.save(model.state_dict(), args.save_dir / "shade_final.pth")
    print(f"Training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
