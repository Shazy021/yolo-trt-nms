from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportConfig:
    """YOLO export configuration.

    mode:
        "engine" — standalone .engine file
        "triton" — Triton model repository
    """
    mode: str = "engine"  # "engine" | "triton"
    triton_model_name: str = "yolo_detector"
    triton_model_version: int = 1

    # common
    model: str = "yolo11l"
    model_path: str | None = None
    height: int = 640
    width: int = 640
    fp16: bool = True
    workspace_gb: int = 8
    output_dir: Path = Path("./export")

    # NMS
    iou_threshold: float = 0.45
    score_threshold: float = 0.3
    max_detections: int = 100

    # batch
    dynamic_batch: bool = True
    min_batch: int = 1
    max_batch: int = 16
    opt_batch: int = 8

    @property
    def model_name(self) -> str:
        return Path(self.model_path).stem if self.model_path else self.model
