"""Step 5: Use the model (load checkpoint and run inference)."""

import argparse
from pathlib import Path
import torch

from step2_lightning_module import Lit3DClassifier


def find_latest_checkpoint(log_dir: Path) -> Path:
    checkpoints = sorted(log_dir.rglob("best_model.pth"))
    if not checkpoints:
        checkpoints = sorted(log_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under: {log_dir}")
    return checkpoints[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Lightning step 5: use model")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--model-name", default="resnet3d_50")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-dir", default="/mnt/pve/work/work-hugo/_runs")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoint is None:
        ckpt = find_latest_checkpoint(Path(args.log_dir))
    else:
        ckpt = Path(args.checkpoint)

    model = Lit3DClassifier.load_from_checkpoint(
        checkpoint_path=str(ckpt),
        model_name=args.model_name,
        lr=args.lr,
    )
    model.eval()

    fake_batch = torch.randn(
        args.batch_size, 3, args.depth, args.height, args.width
    )
    with torch.no_grad():
        logits = model(fake_batch).view(-1)
    print("Predictions (logits):", logits)


if __name__ == "__main__":
    main()
