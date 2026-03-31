"""
YOLO -> TensorRT export.

Usage:
    python export.py                              # engine mode, defaults
    python export.py --mode triton                # triton model repository
    python export.py --model yolo11s --fp16       # custom model
    python export.py --model-path my_model.pt     # custom weights

Modes:
    engine  -- standalone .engine file (local trtexec)
    triton  -- Triton model repository (trtexec in Docker)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from config import ExportConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# =============================================================================
# NMS Plugin
# =============================================================================

class EfficientNMSPlugin(torch.autograd.Function):
    """Wrapper for TRT::EfficientNMS_TRT.

    YOLO outputs probabilities (0-1), not logits, so score_activation=0.
    """

    @staticmethod
    def forward(ctx, boxes, scores, iou_threshold, score_threshold, max_detections):
        B = boxes.shape[0]
        C = scores.shape[-1]
        return (
            torch.randint(0, max_detections, (B, 1), dtype=torch.int32),
            torch.randn(B, max_detections, 4),
            torch.rand(B, max_detections),
            torch.randint(0, C, (B, max_detections), dtype=torch.int32),
        )

    @staticmethod
    def symbolic(g, boxes, scores, iou_threshold, score_threshold, max_detections):
        return g.op(
            "TRT::EfficientNMS_TRT",
            boxes, scores,
            iou_threshold_f=float(iou_threshold),
            score_threshold_f=float(score_threshold),
            max_output_boxes_i=int(max_detections),
            background_class_i=-1,
            box_coding_i=1,
            score_activation_i=0,
            class_agnostic_i=0,
            plugin_version_s="1",
            outputs=4,
        )


# =============================================================================
# Model Wrapper
# =============================================================================

class YOLOWithNMS(nn.Module):
    """YOLO + NMS for ONNX export.

    Input:  [B, 3, H, W]
    Output: num_detections [B, 1], boxes [B, max_det, 4], scores [B, max_det], classes [B, max_det]
    """

    def __init__(self, model: nn.Module, iou_threshold: float, score_threshold: float, max_detections: int):
        super().__init__()
        self.model = model
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]

        boxes = out[:, :4, :].permute(0, 2, 1)   # [B, N, 4]
        scores = out[:, 4:, :].permute(0, 2, 1)  # [B, N, C]

        return EfficientNMSPlugin.apply(
            boxes, scores,
            self.iou_threshold,
            self.score_threshold,
            self.max_detections,
        )


# =============================================================================
# Steps
# =============================================================================

def export_onnx(cfg: ExportConfig) -> Path:
    """YOLO .pt -> ONNX with NMS plugin."""
    from ultralytics import YOLO

    onnx_path = cfg.output_dir / f"{cfg.model_name}.onnx"

    log.info("Loading model: %s", cfg.model_path or cfg.model)
    model = YOLO(cfg.model_path or f"{cfg.model}.pt")

    wrapped = YOLOWithNMS(
        model.model,
        cfg.iou_threshold,
        cfg.score_threshold,
        cfg.max_detections,
    )
    wrapped.eval()

    dynamic_axes = None
    if cfg.dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in
                        ["images", "num_detections", "boxes", "scores", "classes"]}

    dummy = torch.randn(1, 3, cfg.height, cfg.width)
    log.info("Exporting ONNX -> %s", onnx_path)

    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=["images"],
        output_names=["num_detections", "boxes", "scores", "classes"],
        opset_version=17,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        log.info("ONNX validation: OK")
    except ImportError:
        log.warning("onnx not installed, skipping validation")

    return onnx_path


def _trtexec_base(cfg: ExportConfig, onnx_path: str, trtexec: str = "trtexec") -> list[str]:
    """Common trtexec arguments."""
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        "--fp16" if cfg.fp16 else "",
        f"--memPoolSize=workspace:{cfg.workspace_gb * 1024 ** 3}",
    ]

    if cfg.dynamic_batch:
        h, w = cfg.height, cfg.width
        cmd += [
            f"--minShapes=images:{cfg.min_batch}x3x{h}x{w}",
            f"--optShapes=images:{cfg.opt_batch}x3x{h}x{w}",
            f"--maxShapes=images:{cfg.max_batch}x3x{h}x{w}",
        ]

    return [c for c in cmd if c]


def build_engine(cfg: ExportConfig, onnx_path: Path) -> Path:
    """ONNX -> .engine (local trtexec)."""
    engine_path = cfg.output_dir / f"{cfg.model_name}.engine"
    h, w = cfg.height, cfg.width

    cmd = _trtexec_base(cfg, str(onnx_path)) + [f"--saveEngine={engine_path}"]

    if not cfg.dynamic_batch:
        cmd += [f"--shapes=images:1x3x{h}x{w}"]

    log.info("Building engine -> %s", engine_path)
    log.info("Running: %s", " ".join(cmd))

    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        sys.exit(f"trtexec failed with code {r.returncode}")

    log.info("Engine saved: %s", engine_path)
    return engine_path


def build_triton(cfg: ExportConfig, onnx_path: Path) -> Path:
    """ONNX -> Triton model repository (trtexec in Docker)."""
    version_dir = cfg.output_dir / "triton_model_repository" / cfg.triton_model_name / str(cfg.triton_model_version)
    version_dir.mkdir(parents=True, exist_ok=True)

    # config.pbtxt
    max_batch = cfg.max_batch if cfg.dynamic_batch else 1
    config_pbtxt = f"""name: "{cfg.triton_model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch}

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, {cfg.height}, {cfg.width} ]
  }}
]
output [
  {{
    name: "num_detections"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "boxes"
    data_type: TYPE_FP32
    dims: [ {cfg.max_detections}, 4 ]
  }},
  {{
    name: "scores"
    data_type: TYPE_FP32
    dims: [ {cfg.max_detections} ]
  }},
  {{
    name: "classes"
    data_type: TYPE_INT32
    dims: [ {cfg.max_detections} ]
  }}
]
dynamic_batching {{
  preferred_batch_size: [ {cfg.opt_batch} ]
  max_queue_delay_microseconds: 5000
}}
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
    config_path = version_dir.parent / "config.pbtxt"
    config_path.write_text(config_pbtxt)
    log.info("Triton config -> %s", config_path)

    # trtexec in Docker — ONNX path inside container (output_dir -> /export)
    container_onnx = f"/export/{cfg.model_name}.onnx"
    rel = f"triton_model_repository/{cfg.triton_model_name}/{cfg.triton_model_version}"
    container_engine = f"/export/{rel}/model.plan"

    cmd = _trtexec_base(cfg, container_onnx, trtexec="/usr/src/tensorrt/bin/trtexec") + [
        f"--saveEngine={container_engine}",
        "--allowGPUFallback",
    ]

    if not cfg.dynamic_batch:
        h, w = cfg.height, cfg.width
        cmd += [f"--shapes=images:1x3x{h}x{w}"]

    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{cfg.output_dir.absolute()}:/export",
        "nvcr.io/nvidia/tritonserver:26.01-py3",
        "bash", "-c", " ".join(cmd),
    ]

    log.info("Building TensorRT engine in Docker...")
    r = subprocess.run(docker_cmd, text=True)
    if r.returncode != 0:
        sys.exit(f"Docker trtexec failed with code {r.returncode}")

    repo_path = version_dir.parent.parent
    log.info("Triton repository ready -> %s", repo_path)
    return repo_path


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO -> TensorRT export")
    p.add_argument("--mode", choices=["engine", "triton"])
    p.add_argument("--model")
    p.add_argument("--model-path")
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--workspace", type=int)
    p.add_argument("--output")
    p.add_argument("--iou", type=float)
    p.add_argument("--score", type=float)
    p.add_argument("--max-det", type=int)
    p.add_argument("--static-batch", action="store_true")
    p.add_argument("--min-batch", type=int)
    p.add_argument("--max-batch", type=int)
    p.add_argument("--opt-batch", type=int)
    p.add_argument("--triton-name")
    p.add_argument("--triton-version", type=int)
    return p.parse_args()


def main():
    args = parse_args()

    # only user-provided values — don't overwrite ExportConfig defaults
    overrides = {}
    if args.mode is not None:
        overrides["mode"] = args.mode
    if args.model is not None:
        overrides["model"] = args.model
    if args.model_path is not None:
        overrides["model_path"] = args.model_path
    if args.height is not None:
        overrides["height"] = args.height
    if args.width is not None:
        overrides["width"] = args.width
    if args.fp32:
        overrides["fp16"] = False
    if args.workspace is not None:
        overrides["workspace_gb"] = args.workspace
    if args.output is not None:
        overrides["output_dir"] = Path(args.output)
    if args.iou is not None:
        overrides["iou_threshold"] = args.iou
    if args.score is not None:
        overrides["score_threshold"] = args.score
    if args.max_det is not None:
        overrides["max_detections"] = args.max_det
    if args.static_batch:
        overrides["dynamic_batch"] = False
    if args.min_batch is not None:
        overrides["min_batch"] = args.min_batch
    if args.max_batch is not None:
        overrides["max_batch"] = args.max_batch
    if args.opt_batch is not None:
        overrides["opt_batch"] = args.opt_batch
    if args.triton_name is not None:
        overrides["triton_model_name"] = args.triton_name
    if args.triton_version is not None:
        overrides["triton_model_version"] = args.triton_version

    cfg = ExportConfig(**overrides)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Mode: %s | Model: %s | %s", cfg.mode.upper(), cfg.model_path or cfg.model,
             "FP16" if cfg.fp16 else "FP32")

    onnx_path = export_onnx(cfg)

    if cfg.mode == "engine":
        build_engine(cfg, onnx_path)
    else:
        build_triton(cfg, onnx_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
