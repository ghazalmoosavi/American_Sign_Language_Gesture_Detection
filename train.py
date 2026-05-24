import os
import yaml
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

try:
    import wandb  
except ImportError:
    wandb = None


DEFAULT_CONFIG = {
    # Model
    "model": "yolov8n.pt",
    "task": "detect",

    # Dataset
    "data": "configs/asl.yaml",

    # Training hyperparams
    "epochs": 100,
    "patience": 20,
    "batch": 16,
    "imgsz": 640,
    "workers": 8,
    "device": "auto",   # auto / cpu / mps / 0 / 0,1
    "amp": True,

    # Optimizer
    "optimizer": "AdamW",
    "lr0": 1e-3,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 5e-4,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # Loss weights
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # Augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 15.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 5.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.0,
    "erasing": 0.4,

    # Output
    "project": "runs/asl",
    "name": "yolov8_ciou",
    "save": True,
    "save_period": 10,
    "val": True,
    "plots": True,
    "verbose": True,
}


def resolve_device(device_arg: str = "auto") -> str:

    if device_arg and device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def build_dataset_yaml(dataset_root: str, output_path: str = "configs/asl.yaml"):
    """
    Auto-generate dataset YAML from a local YOLOv8 dataset export.
    Expected structure:
        dataset_root/
            train/images/
            train/labels/
            valid/images/
            valid/labels/
            test/images/
            test/labels/
            data.yaml
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    src_yaml = dataset_root / "data.yaml"

    if src_yaml.exists():
        with open(src_yaml, "r") as f:
            meta = yaml.safe_load(f)

        meta["path"] = str(dataset_root)
        meta["train"] = "train/images"
        meta["val"] = "valid/images"

        if (dataset_root / "test").exists():
            meta["test"] = "test/images"

        os.makedirs("configs", exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        LOGGER.info(f"Dataset YAML written → {output_path}")
        LOGGER.info(f"  Classes ({meta.get('nc', 'unknown')}): {meta.get('names', [])}")
        return output_path

    raise FileNotFoundError(f"data.yaml not found in {dataset_root}")


def train(cfg: dict, dataset_root: str | None = None, use_wandb: bool = False):
    
    if dataset_root:
        cfg["data"] = build_dataset_yaml(dataset_root)

    
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    LOGGER.info(f"Using device: {cfg['device']}")

    
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        wandb.init(project="asl-detection", config=cfg)

    
    model = YOLO(cfg.pop("model"))

    
    LOGGER.info("Starting ASL training with CIoU loss ...")
    results = model.train(**cfg)

    
    metrics = results.results_dict
    LOGGER.info("\n" + "─" * 50)
    LOGGER.info("Training complete!")
    LOGGER.info(f"  mAP@0.5      : {metrics.get('metrics/mAP50(B)', 0):.4f}")
    LOGGER.info(f"  mAP@0.5:0.95 : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    LOGGER.info(f"  Precision    : {metrics.get('metrics/precision(B)', 0):.4f}")
    LOGGER.info(f"  Recall       : {metrics.get('metrics/recall(B)', 0):.4f}")
    LOGGER.info("─" * 50)

    if use_wandb and wandb is not None:
        wandb.log(metrics)
        wandb.finish()

    return model, results


def resume(checkpoint: str, extra_epochs: int = 50, device: str = "auto"):
    """Resume training from a saved checkpoint."""
    model = YOLO(checkpoint)
    results = model.train(resume=True, epochs=extra_epochs, device=resolve_device(device))
    return model, results




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for ASL detection")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path.home() / "Desktop" / "ML" / "ASL Letters.v5i.yolov8"),
        help="Path to local YOLOv8 dataset root",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, mps, 0, 0,1, etc.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    if args.resume:
        resume(args.resume, device=args.device)
    else:
        cfg = DEFAULT_CONFIG.copy()
        cfg.update({
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
        })
        train(cfg, dataset_root=args.dataset, use_wandb=args.wandb)
