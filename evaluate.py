"""
ASL Model Evaluation
Generates per-class precision/recall/F1, confusion matrix, and speed benchmarks.
"""
 
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
 
 
# ──────────────────────────────────────────────
# Auto-detect best weights
# ──────────────────────────────────────────────
 
def find_best_weights(start: Path = Path(".")) -> str | None:
    candidates = sorted(start.rglob("best.pt"))
    if candidates:
        print(f"Auto-detected weights: {candidates[0]}")
        return str(candidates[0])
    return None
 
 
def find_data_yaml(start: Path = Path(".")) -> str | None:
    for p in [Path("configs/asl.yaml"), Path("data/asl/data.yaml")]:
        if p.exists():
            print(f"Auto-detected data yaml: {p}")
            return str(p)
    candidates = sorted(start.rglob("*.yaml"))
    for c in candidates:
        if "asl" in c.name.lower() or "data" in c.name.lower():
            print(f"Auto-detected data yaml: {c}")
            return str(c)
    return None
 
 
# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
 
def evaluate(weights: str, data_yaml: str, imgsz: int = 640,
             device: str = "cpu", half: bool = False, split: str = "val"):
    model = YOLO(weights)
 
    print(f"Evaluating {weights} on {split} split …")
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=16,
        device=device,
        half=half and device != "cpu",
        split=split,
        plots=True,
        save_json=True,
        verbose=False,
    )
 
    rd = metrics.results_dict
    summary = {
        "mAP@0.5":      round(rd.get("metrics/mAP50(B)",    0), 4),
        "mAP@0.5:0.95": round(rd.get("metrics/mAP50-95(B)", 0), 4),
        "precision":    round(rd.get("metrics/precision(B)", 0), 4),
        "recall":       round(rd.get("metrics/recall(B)",    0), 4),
        "fitness":      round(rd.get("metrics/fitness",      0), 4),
    }
 
    print("\n" + "═" * 40)
    print("  EVALUATION RESULTS")
    print("═" * 40)
    for k, v in summary.items():
        bar = "█" * int(v * 30)
        print(f"  {k:<18} {v:.4f}  {bar}")
    print("═" * 40)
 
    out_dir = Path("runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {out_dir / 'summary.json'}")
 
    return metrics, summary
 
 
def speed_benchmark(weights: str, imgsz: int = 640, device: str = "cpu",
                    n_warmup: int = 10, n_bench: int = 100):
    """Benchmark inference latency and FPS."""
    import torch
    model = YOLO(weights)
    model.model.eval()
 
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    if device != "cpu":
        dummy = dummy.to(device)
 
    print(f"\nWarming up ({n_warmup} frames) …")
    for _ in range(n_warmup):
        model.predict(dummy, verbose=False, device=device)
 
    print(f"Benchmarking ({n_bench} frames) …")
    t0 = time.perf_counter()
    for _ in range(n_bench):
        model.predict(dummy, verbose=False, device=device)
    elapsed = time.perf_counter() - t0
 
    ms_per_frame = elapsed / n_bench * 1000
    fps = n_bench / elapsed
 
    print(f"\n⚡ Speed Benchmark ({n_bench} frames, device={device})")
    print(f"   Latency : {ms_per_frame:.2f} ms/frame")
    print(f"   FPS     : {fps:.1f}")
    print(f"   {'✅ Real-time capable (≥30 FPS)' if fps >= 30 else '⚠️  Below 30 FPS — consider GPU for real-time use'}")
 
    return {"ms_per_frame": round(ms_per_frame, 2), "fps": round(fps, 1)}
 
 
def plot_per_class(metrics, class_names: list, out_dir: str = "runs/eval"):
    """Bar chart of per-class AP@0.5."""
    ap_per_class = metrics.box.ap50
    if ap_per_class is None or len(ap_per_class) == 0:
        print("Per-class AP not available.")
        return
 
    names = class_names[:len(ap_per_class)]
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ["#00DC6E" if v >= 0.90 else "#00B4FF" if v >= 0.70 else "#FF4444"
              for v in ap_per_class]
    ax.bar(names, ap_per_class, color=colors, edgecolor="none", width=0.7)
    ax.axhline(0.92, ls="--", lw=1.2, color="#FFFFFF44", label="Target mAP 0.92")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ASL Class", fontsize=11)
    ax.set_ylabel("AP@0.5", fontsize=11)
    ax.set_title("Per-Class AP@0.5 — ASL Detection", fontsize=13, pad=14)
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#1A1A1A")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
        spine.set_linewidth(0.5)
    plt.xticks(rotation=45, ha="right", color="white", fontsize=9)
    plt.yticks(color="white")
    plt.tight_layout()
    out = Path(out_dir) / "per_class_ap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Per-class AP chart → {out}")
    plt.close()
 
 
# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASL YOLOv8 model")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to best.pt (auto-detected if omitted)")
    parser.add_argument("--data",    type=str, default=None,
                        help="Path to dataset YAML (auto-detected if omitted)")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--device",  type=str, default="cpu",
                        help="Device: cpu, 0 (GPU), mps (Apple Silicon)")
    parser.add_argument("--split",   type=str, default="val",
                        choices=["val", "test"])
    parser.add_argument("--benchmark", action="store_true",
                        help="Also run FPS speed benchmark")
    args = parser.parse_args()
 
    # Auto-detect if not provided
    weights = args.weights or find_best_weights()
    data    = args.data    or find_data_yaml()
 
    if not weights:
        raise FileNotFoundError(
            "Could not find best.pt. Pass it explicitly: --weights path/to/best.pt"
        )
    if not data:
        raise FileNotFoundError(
            "Could not find dataset YAML. Pass it explicitly: --data configs/asl.yaml"
        )
 
    metrics, summary = evaluate(
        weights=weights,
        data_yaml=data,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
    )
 
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
    plot_per_class(metrics, class_names)
 
    if args.benchmark:
        speed_benchmark(weights, imgsz=args.imgsz, device=args.device)
 