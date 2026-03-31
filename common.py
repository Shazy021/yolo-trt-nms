"""Shared utilities for YOLO inference (local + Triton)."""

import cv2
import numpy as np
from typing import Tuple, Optional

# =============================================================================
# COCO
# =============================================================================

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]

np.random.seed(42)
CLASS_COLORS = np.random.randint(50, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)

# reusable buffer to avoid allocations
_canvas = None


# =============================================================================
# Preprocess
# =============================================================================

def preprocess(frame: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, Tuple[int, int, float, int, int]]:
    """Resize + pad + normalize (letterbox).

    Returns:
        tensor:   (C, H, W) float32, RGB, [0..1]
        pad_info: (pad_x, pad_y, scale, orig_h, orig_w)
    """
    global _canvas
    if _canvas is None or _canvas.shape != (h, w, 3):
        _canvas = np.full((h, w, 3), 114, dtype=np.uint8)

    orig_h, orig_w = frame.shape[:2]
    scale = min(w / orig_w, h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    interp = cv2.INTER_AREA if new_w < orig_w else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    pad_x = (w - new_w) // 2
    pad_y = (h - new_h) // 2

    _canvas.fill(114)
    _canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    tensor = _canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor, (pad_x, pad_y, scale, orig_h, orig_w)


# =============================================================================
# Scale boxes back to original image
# =============================================================================

def scale_boxes(boxes: np.ndarray, pad_info: Tuple) -> np.ndarray:
    """Model coordinates -> original image coordinates."""
    pad_x, pad_y, scale, orig_h, orig_w = pad_info
    out = boxes.astype(np.float64).copy()
    out[:, 0] = (out[:, 0] - pad_x) / scale
    out[:, 1] = (out[:, 1] - pad_y) / scale
    out[:, 2] = (out[:, 2] - pad_x) / scale
    out[:, 3] = (out[:, 3] - pad_y) / scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out.astype(np.int32)


# =============================================================================
# Draw
# =============================================================================

def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    conf: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    out = frame.copy()
    if len(boxes) == 0:
        return out

    for i in range(len(boxes)):
        if scores[i] < conf:
            continue

        x1, y1, x2, y2 = int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])
        color = tuple(int(c) for c in CLASS_COLORS[int(classes[i]) % len(CLASS_COLORS)])

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{COCO_CLASSES[int(classes[i])]} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out


def draw_stats(
    frame: np.ndarray,
    fps: float,
    num_det: int,
    mode: str = "LOCAL",
    batch: int = 1,
    infer_ms: Optional[float] = None,
) -> np.ndarray:
    """Draw stats HUD overlay at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 42

    # semi-transparent bar
    overlay = np.full((bar_h, w, 3), 30, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.7, frame[:bar_h], 0.3, 0, frame[:bar_h])

    # separator line
    cv2.line(frame, (0, bar_h), (w, bar_h), (80, 200, 80), 1)

    # columns
    col1 = f"FPS: {fps:.1f}"
    col2 = f"Det: {num_det}"
    col3 = f"Batch: {batch}"
    col4 = f"Mode: {mode}"
    if infer_ms is not None:
        col5 = f"Infer: {infer_ms:.1f}ms"
        text = f"  {col1}    {col2}    {col3}    {col4}    {col5}"
    else:
        text = f"  {col1}    {col2}    {col3}    {col4}"

    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

    return frame


# =============================================================================
# Video writer
# =============================================================================

def make_writer(output: str, fps: int, size: tuple):
    """Create video writer via ffmpeg. Prefers NVENC (GPU), falls back to libx264."""
    W, H = size
    import shutil, subprocess

    # find ffmpeg: system PATH or imageio-ffmpeg bundle
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except ImportError:
            pass

    if ffmpeg:
        # try NVENC first (GPU encoding), then libx264 (CPU)
        for codec, preset, crf in [
            ("h264_nvenc", "p4", "23"),
            ("libx264", "ultrafast", "23"),
        ]:
            cmd = [
                ffmpeg, "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{W}x{H}", "-r", str(fps),
                "-i", "-",
                "-c:v", codec, "-preset", preset, "-crf", crf,
                "-pix_fmt", "yuv420p",
                output,
            ]
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                # wait for ffmpeg to init codec
                import time; time.sleep(0.3)
                if proc.poll() is None:
                    class _FFmpegWriter:
                        def __init__(self, p):
                            self.proc = p
                        def write(self, frame):
                            self.proc.stdin.write(frame.tobytes())
                        def release(self):
                            self.proc.stdin.close()
                            self.proc.wait()
                    return _FFmpegWriter(proc)
                proc.wait()
            except Exception:
                continue

    # fallback: mp4v (big files, no extra deps)
    return cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
