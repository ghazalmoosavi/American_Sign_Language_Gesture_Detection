# ASL Gesture Detection with YOLOv8

A real-time American Sign Language (ASL) gesture detection system built with **YOLOv8**. The project is designed for robust alphabet recognition, low-latency inference, and practical live transcription from webcam, video, or image input.

## Project Overview

This repository implements an end-to-end ASL detection pipeline with a focus on production-style structure and measurable performance. It includes:

* dataset preparation in YOLO format
* model training with tuned optimization and augmentation settings
* validation and per-class performance analysis
* real-time inference with temporal smoothing and text building
* optional checkpointing, plotting, and experiment tracking

The goal is not only to detect ASL signs, but to make the output usable in real time by reducing prediction noise and converting repeated detections into readable text.

## Model Overview

The detector is trained on 26 ASL alphabet classes:

```text
0  A    9  J   18 S
1  B   10  K   19 T
2  C   11  L   20 U
3  D   12  M   21 V
4  E   13  N   22 W
5  F   14  O   23 X
6  G   15  P   24 Y
7  H   16  Q   25 Z
8  I   17  R
```

The model outputs bounding boxes, class labels, and confidence scores for each detected sign.

## Training Strategy

The training pipeline is tuned to balance three priorities: accuracy, stability, and inference speed.

### Base model

* `model: yolov8n.pt`

A lightweight YOLOv8 variant is used as the baseline to keep the system efficient enough for real-time use. This choice is especially important for webcam inference and deployment on modest hardware.

### Optimization settings

* `optimizer: AdamW` — stable optimization with good generalization behavior
* `lr0: 1e-3` — controlled initial learning rate for custom detection training
* `lrf: 0.01` — gradual learning-rate decay toward the end of training
* `momentum: 0.937` — smoother parameter updates
* `weight_decay: 5e-4` — regularization to reduce overfitting
* `warmup_epochs: 3` — gentle ramp-up at the beginning of training
* `warmup_momentum: 0.8` — stabilizes early optimization steps
* `warmup_bias_lr: 0.1` — helps the model adapt quickly during warmup

### Loss weighting

* `box: 7.5` — emphasizes precise bounding-box regression
* `cls: 0.5` — classification loss weight
* `dfl: 1.5` — improves localization quality through distribution-based regression

These values are useful for gesture detection because small spatial differences matter when distinguishing similar hand poses.

### Augmentation policy

The augmentation setup is intentionally strong enough to improve robustness while remaining realistic for ASL gestures:

* HSV shifts for lighting variation
* rotation, translation, scaling, and shear for viewpoint variation
* horizontal flipping for mirrored camera conditions
* mosaic and mixup for generalization
* erasing for partial occlusion tolerance

### Runtime configuration

* `imgsz: 640` — detection resolution
* `batch: 16` — training throughput and memory balance
* `epochs: 100` — sufficient budget for convergence
* `patience: 20` — early stopping when validation plateaus
* `device: auto` — CUDA, MPS, or CPU selection depending on hardware

## Why This Setup Works

This configuration is suited to ASL recognition because it keeps the model compact while still allowing enough capacity for accurate hand detection. The augmentation strategy improves resilience to camera motion, lighting changes, and hand placement differences. The heavier bounding-box weighting helps the detector focus on localization quality, which is critical when multiple ASL signs look visually similar.

## Evaluation Results

Example validation results from the trained model:

* **mAP@0.5:** 0.9873
* **mAP@0.5:0.95:** 0.7364
* **Precision:** 0.9721
* **Recall:** 0.9590

These results show that the model performs strongly at standard detection thresholds and remains reliable under a stricter evaluation setting as well.

### Evaluation outputs

The evaluation script reports:

* mAP@0.5
* mAP@0.5:0.95
* precision
* recall
* fitness score
* per-class AP visualization
* optional FPS and latency benchmark

## Inference Pipeline

The inference engine is designed for live interaction and follows a clear processing flow:

1. load trained weights
2. open a webcam, video file, or image source
3. run YOLOv8 prediction on each frame
4. filter detections using confidence and IoU thresholds
5. draw bounding boxes, labels, and confidence scores
6. stabilize predictions across frames
7. convert repeated gestures into words
8. display the final text and performance overlay

### Temporal smoothing

Real-time gesture recognition is noisy if every frame is accepted immediately. To address this, the inference engine uses a frame-hold mechanism: a label is only committed after it appears consistently across multiple frames. This reduces flicker and prevents accidental character insertion.

### Control tokens

The model includes control labels that make transcription more practical:

* `nothing` resets the current gesture state
* `space` commits the current word and starts a new one
* `del` removes the last character from the current word

### Visual feedback

Confidence-based box coloring provides immediate visual feedback during inference:

* high confidence → green
* medium confidence → yellow/orange
* low confidence → red

FPS is also shown during runtime so the model’s real-time suitability can be monitored directly.

## Inference Controls

* **Q** — quit
* **C** — clear the current word buffer
* **S** — save a screenshot

## Key Files

* `prepare_dataset.py` — dataset configuration and split statistics
* `train.py` — training pipeline and hyperparameter configuration
* `evaluate.py` — validation metrics, plots, and benchmark utilities
* `inference.py` — live webcam/video inference with word building

## Project Strengths

* compact model design for real-time performance
* practical live transcription logic
* strong validation metrics
* structured training and evaluation workflow
* flexible device support, including Apple Silicon via MPS
* automatic checkpoint detection for easier reuse


