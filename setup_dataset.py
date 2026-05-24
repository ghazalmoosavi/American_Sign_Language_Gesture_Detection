import os
import argparse
from pathlib import Path

import yaml


def prepare_config(dataset_root: str, output_yaml: str = "configs/asl.yaml"):
   
    dataset_root = Path(dataset_root).expanduser().resolve()
    src = dataset_root / "data.yaml"

    if not src.exists():
        raise FileNotFoundError(f"No data.yaml found in: {dataset_root}")

    with open(src, "r") as f:
        meta = yaml.safe_load(f)

    meta["path"] = str(dataset_root)
    meta["train"] = "train/images"
    meta["val"] = "valid/images"

    if (dataset_root / "test").exists():
        meta["test"] = "test/images"

    os.makedirs("configs", exist_ok=True)
    with open(output_yaml, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f"Config written → {output_yaml}")
    print(f"Classes ({meta.get('nc', 'unknown')}): {meta.get('names', [])}")
    return output_yaml


def dataset_stats(dataset_root: str):
    """
    Print image/label counts per split.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()

    print("\n── Dataset Statistics ─────────────────")
    total_imgs = 0
    total_lbls = 0

    for split in ("train", "valid", "test"):
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"

        if not img_dir.exists():
            continue

        n_imgs = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        print(f"  {split:6s}  images={n_imgs:5d}  labels={n_lbls:5d}")
        total_imgs += n_imgs
        total_lbls += n_lbls

    print(f"  {'total':6s}  images={total_imgs:5d}  labels={total_lbls:5d}")
    print("────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare local ASL dataset")
    parser.add_argument(
        "--dataset-root",
        default=str(Path.home() / "Desktop" / "ML" / "ASL Letters.v5i.yolov8"),
        help="Path to your local dataset folder",
    )
    parser.add_argument("--output", default="configs/asl.yaml")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print dataset statistics",
    )

    args = parser.parse_args()

    if args.stats_only:
        dataset_stats(args.dataset_root)
    else:
        prepare_config(args.dataset_root, args.output)
        dataset_stats(args.dataset_root)
