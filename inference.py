"""
ASL Real-Time Inference Engine
Webcam · Video file · Image
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import torch

# ──────────────────────────────────────────────
# ASL alphabet + blank
# ──────────────────────────────────────────────
ASL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

CONF_COLORS = {
    "high":   (0, 220, 110),
    "medium": (0, 180, 255),
    "low":    (60, 60, 255),
}


def conf_color(score: float) -> tuple:
    if score >= 0.80:
        return CONF_COLORS["high"]
    if score >= 0.50:
        return CONF_COLORS["medium"]
    return CONF_COLORS["low"]


def resolve_device(device_arg: str = "auto") -> str:
    """Choose CUDA → MPS → CPU automatically."""
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_best_weights(start: Path = Path(".")) -> str | None:
    """Search for best.pt anywhere under the project folder."""
    candidates = sorted(start.rglob("best.pt"))
    if candidates:
        print(f"Auto-detected weights: {candidates[0]}")
        return str(candidates[0])
    return None


# ──────────────────────────────────────────────
# FPS tracker
# ──────────────────────────────────────────────

class FPSMeter:
    def __init__(self, window: int = 30):
        self._times = deque(maxlen=window)
        self._t0 = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._t0)
        self._t0 = now
        if len(self._times) < 2:
            return 0.0
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0


# ──────────────────────────────────────────────
# Letter buffer → word builder
# ──────────────────────────────────────────────

class WordBuilder:
    def __init__(self, hold_frames: int = 12, max_word: int = 20):
        self.hold_frames = hold_frames
        self._current = None
        self._count = 0
        self.word = ""
        self.sentence = []
        self.max_word = max_word

    def update(self, label):
        if label is None or label == "nothing":
            self._current = None
            self._count = 0
            return None

        if label == "space":
            if self.word:
                self.sentence.append(self.word)
                self.word = ""
            self._current = None
            self._count = 0
            return None

        if label == "del":
            if self.word:
                self.word = self.word[:-1]
            self._current = None
            self._count = 0
            return None

        if label == self._current:
            self._count += 1
        else:
            self._current = label
            self._count = 1

        if self._count == self.hold_frames:
            self.word = (self.word + label)[-self.max_word:]
            self._count = 0
            return label
        return None

    @property
    def display(self) -> str:
        parts = self.sentence[-3:] + ([self.word] if self.word else [])
        return " ".join(parts)

    def clear(self):
        self.word = ""
        self.sentence = []
        self._current = None
        self._count = 0


# ──────────────────────────────────────────────
# HUD drawing helpers
# ──────────────────────────────────────────────

def draw_hud(frame, fps, word_builder, model_name):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 44), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    fps_color = (0, 220, 110) if fps >= 28 else (0, 165, 255) if fps >= 15 else (60, 60, 220)
    cv2.putText(frame, f"FPS {fps:5.1f}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, model_name, (w - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    panel_h = 56
    panel_y = h - panel_h
    cv2.rectangle(frame, (0, panel_y), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(frame, 0.80, overlay, 0.20, 0, frame)

    text = word_builder.display or "..."
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.90, 1)
    x = min(12, w - tw - 12)
    cv2.putText(frame, text, (x, panel_y + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.90, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q quit  C clear  S screenshot", (12, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    return frame


def draw_detections(frame, result, class_names):
    top_label = None
    top_conf = 0.0

    if result.boxes is None:
        return None

    for box in result.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        if conf > top_conf:
            top_conf = conf
            top_label = label

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = conf_color(conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        tag = f"{label}  {conf:.0%}"
        (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        pill_y = max(y1 - th - 10, 0)
        cv2.rectangle(frame, (x1, pill_y), (x1 + tw + 8, pill_y + th + 8), color, -1)
        cv2.putText(frame, tag, (x1 + 4, pill_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2, cv2.LINE_AA)

    return top_label


# ──────────────────────────────────────────────
# Main inference loop
# ──────────────────────────────────────────────

def run_inference(
    weights: str,
    source=0,
    conf_thresh: float = 0.50,
    iou_thresh: float = 0.45,
    imgsz: int = 640,
    device: str = "auto",
    half: bool = False,
    save_video: bool = False,
    hold_frames: int = 12,
):
    device = resolve_device(device)

    model = YOLO(weights)
    class_names = model.names
    names_list = [class_names[i] for i in range(len(class_names))]
    model_tag = Path(weights).stem

    print(f"Model  : {weights}")
    print(f"Device : {device}  |  Half: {half and device == '0'}  |  conf≥{conf_thresh}")
    print(f"Classes ({len(names_list)}): {names_list}")

    is_webcam = isinstance(source, int) or str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if save_video:
        out_path = "asl_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (W, H))
        print(f"Saving output → {out_path}")

    fps_meter = FPSMeter()
    word_builder = WordBuilder(hold_frames=hold_frames)
    screenshot_n = 0

    print("Press  Q=quit  C=clear word  S=screenshot")

    for result in model.predict(
        source=int(source) if str(source).isdigit() else source,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        device=device,
        half=half and device == "0",
        stream=True,
        verbose=False,
    ):
        frame = result.orig_img.copy()

        top_label = draw_detections(frame, result, names_list)
        word_builder.update(top_label)
        fps = fps_meter.tick()
        frame = draw_hud(frame, fps, word_builder, model_tag)

        if writer:
            writer.write(frame)

        cv2.imshow("ASL Detection - YOLOv8", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            word_builder.clear()
            print("Word cleared.")
        elif key == ord("s"):
            fn = f"screenshot_{screenshot_n:04d}.png"
            cv2.imwrite(fn, frame)
            screenshot_n += 1
            print(f"Saved {fn}")

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nFinal sentence: {word_builder.display}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time ASL inference with YOLOv8")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to best.pt (auto-detected if omitted)")
    parser.add_argument("--source", default="0",
                        help="Webcam index (0) or video/image path")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou",  type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cpu, mps, 0 (CUDA)")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--hold-frames", type=int, default=12,
                        help="Frames a sign must be stable before committing")
    args = parser.parse_args()

    # Auto-detect weights if not provided
    weights = args.weights or find_best_weights()
    if not weights:
        raise FileNotFoundError(
            "Could not find best.pt automatically. "
            "Pass it explicitly: --weights path/to/best.pt"
        )

    run_inference(
        weights=weights,
        source=args.source,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        half=not args.no_half,
        save_video=args.save_video,
        hold_frames=args.hold_frames,
    )