# YOLO TensorRT + Triton Inference Server

YOLO to TensorRT export and high-performance inference with built-in NMS.

Two modes: standalone `.engine` file and NVIDIA Triton Inference Server. Threading pipeline with batching for maximum GPU utilization.

## Demo

<table>
  <tr>
    <th align="center">Local</th>
    <th align="center">Triton</th>
  </tr>
  <tr>
    <td align="center">
      <video width="100%" controls>
        <source src="./assets/moto1_local.mp4" type="video/mp4">
      </video><br>
    </td>
    <td align="center">
      <video width="100%" controls>
        <source src="./assets/moto1_triton.mp4" type="video/mp4">
      </video><br>
    </td>
  </tr>
  <tr>
    <td align="center">
      <video width="100%" controls>
        <source src="./assets/moto2_local.mp4" type="video/mp4">
      </video><br>
    </td>
    <td align="center">
      <video width="100%" controls>
        <source src="./assets/moto2_triton.mp4" type="video/mp4">
      </video><br>
    </td>
  </tr>
</table>

## Threading Pipeline

```
[Reader] ──► [Preprocessor] ──► [GPU Inference] ──► [Renderer]
  Thread       Thread              Thread            Thread
```

- **Reader** — reads frames from video / camera
- **Preprocessor** — resize + pad + normalize, assembles batches
- **GPU Inference** — TensorRT (local) or Triton gRPC
- **Renderer** — draws bounding boxes and FPS stats

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for Triton mode)
- Python 3.10+

## Installation

```bash
# 1. PyTorch with CUDA (pick your version at https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. TensorRT + PyCUDA (versions must match your CUDA)
pip install tensorrt pycuda

# 3. ONNX Runtime (versions must match your CUDA)
# pip install coloredlogs flatbuffers numpy packaging protobuf sympy
# pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu

# 4. Project dependencies
pip install -r requirements.txt
```

## Usage

### Export

```bash
# Standalone .engine (default)
python export.py
python export.py --model yolo11s --fp16

# Triton model repository (Docker)
python export.py --mode triton
python export.py --mode triton --model yolo11n --max-batch 32

# Custom weights
python export.py --model-path my_custom_yolo.pt --mode triton
```

| Parameter | Description | Default |
|---|---|---|
| `--mode` | `engine` or `triton` | `engine` |
| `--model` | Model (yolo11n/s/m/l/x) | `yolo11m` |
| `--model-path` | Path to custom weights | — |
| `--height` / `--width` | Input resolution | 640 |
| `--fp32` | FP32 instead of FP16 | FP16 |
| `--iou` | NMS IoU threshold | 0.45 |
| `--score` | NMS score threshold | 0.5 |
| `--max-det` | Max detections per image | 100 |
| `--static-batch` | Disable dynamic batching | dynamic |
| `--min-batch` / `--max-batch` / `--opt-batch` | Batch sizes | 1 / 16 / 4 |
| `--workspace` | TRT workspace in GB | 8 |
| `--output` | Output directory | `./export` |

### Local Inference

```bash
# Video
python inference_local.py --model export/yolo11m.engine --source video.mp4
python inference_local.py --model export/yolo11m.engine --source video.mp4 --batch 8 --conf 0.3

# Camera
python inference_local.py --model export/yolo11m.engine --source 0

# Image folder
python inference_local.py --model export/yolo11m.engine --source images/

# Save result
python inference_local.py --model export/yolo11m.engine --source video.mp4 --output result.mp4
```

| Parameter | Description | Default |
|---|---|---|
| `--source` | Video, camera (0,1...), or RTSP stream | required |
| `--url` | Triton gRPC URL | localhost:8001 |
| `--model` | Model name on Triton | yolo_detector |
| `--conf` | Min confidence to draw | 0.5 |
| `--batch` | Client-side batch size (1=disabled) | 1 |
| `--output` | Output video file path | — |
| `--no-show` | Disable display window | show |


### Triton Inference

```bash
# Start Triton server
docker compose up -d triton

# Check model is loaded
curl http://localhost:8000/v2/models/yolo_detector

# Run inference — video
python inference_triton.py --source video.mp4
python inference_triton.py --source video.mp4 --batch 8 --conf 0.3

# Run inference — webcam / RTSP
python inference_triton.py --source 0
python inference_triton.py --source rtsp://192.168.1.100:554/stream

# Logs
docker compose logs -f triton

# Stop
docker compose down
```

| Parameter | Description | Default |
|---|---|---|
| `--source` | Video, camera (0,1...), or RTSP stream | required |
| `--url` | Triton gRPC URL | localhost:8001 |
| `--model` | Model name on Triton | yolo_detector |
| `--conf` | Min confidence to draw | 0.5 |
| `--batch` | Client-side batch size (1=disabled) | 1 |
| `--no-show` | Disable display window | show |

### NMS Plugin

Efficient NMS is embedded into the graph via the `TRT::EfficientNMS_TRT` TensorRT plugin. NMS runs on the GPU as part of the engine — no CPU post-processing needed.

NMS parameters are set at export time and baked into the model:

```bash
python export.py --iou 0.5 --score 0.3 --max-det 200
```

## Project Structure

```
.
├── config.py              # Export config (dataclass)
├── common.py              # Shared utilities (COCO, preprocess, draw)
├── export.py              # YOLO → ONNX → TensorRT
├── inference_local.py     # Local inference (.engine)
├── inference_triton.py    # Triton client (gRPC)
├── docker-compose.yml      # Triton server
├── requirements.txt
└── README.md
```

After export:

```
export/
├── yolo11m.onnx                    # engine mode
├── yolo11m.engine                  # engine mode
└── triton_model_repository/        # triton mode
    └── yolo_detector/
        ├── config.pbtxt
        └── 1/
            └── model.plan
```

## Key Technologies

- **TensorRT** — graph optimization, FP16, dynamic batching
- **EfficientNMS_TRT** — GPU NMS plugin (no CPU post-processing)
- **PyCUDA** — direct GPU memory management for local inference
- **NVIDIA Triton** — production inference server with model repository
- **async gRPC** — non-blocking Triton requests with client-side batching
