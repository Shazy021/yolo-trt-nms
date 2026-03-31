"""Local YOLO TensorRT inference (.engine file).

Usage:
    python inference_local.py --model model.engine --source video.mp4
    python inference_local.py --model model.engine --source video.mp4 --batch 4
    python inference_local.py --model model.engine --source images/
"""

import argparse
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Optional

import cv2
import numpy as np

from common import preprocess, scale_boxes, draw_detections, draw_stats, make_writer


# =============================================================================
# TensorRT Engine
# =============================================================================

class TensorRTEngine:
    def __init__(self, path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.cuda = cuda
        self.trt = trt
        logger = trt.Logger(trt.Logger.Severity.WARNING)
        trt.init_libnvinfer_plugins(logger, "")

        print(f"Loading: {path}")
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if not self.engine:
            raise RuntimeError("Failed to load engine")
        self.context = self.engine.create_execution_context()

        # parse tensors
        self.input_name = None
        self.height = 640
        self.width = 640
        self.is_dynamic = False
        self.max_batch = 1
        self.max_detections = 100

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                if len(shape) == 4:
                    self.height, self.width = int(shape[2]), int(shape[3])
                if shape[0] == -1:
                    self.is_dynamic = True
                    self.max_batch = 16
                else:
                    self.max_batch = int(shape[0])

            # max_det from boxes shape
            if name == "boxes" and len(shape) >= 2 and shape[1] != -1:
                self.max_detections = int(shape[1])

        print(f"  {self.height}x{self.width}  max_det={self.max_detections}  "
              f"batch={'dynamic' if self.is_dynamic else self.max_batch}")

        self.stream = cuda.Stream()
        self._alloc_buffers(self.max_batch)

    def _alloc_buffers(self, batch_size: int):
        """Allocate GPU buffers for max_batch."""
        md = self.max_detections

        # input
        inp_size = batch_size * 3 * self.height * self.width
        self.inp_host = self.cuda.pagelocked_empty(inp_size, dtype=np.float32)
        self.inp_dev = self.cuda.mem_alloc(self.inp_host.nbytes)

        # outputs
        self.out_host = {}
        self.out_dev = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                continue
            dtype = self.engine.get_tensor_dtype(name)
            np_dt = np.int32 if dtype == self.trt.DataType.INT32 else np.float32

            if name == "boxes":
                size = batch_size * md * 4
            elif name == "num_detections":
                size = batch_size
            else:  # scores, classes
                size = batch_size * md

            self.out_host[name] = self.cuda.pagelocked_empty(size, dtype=np_dt)
            self.out_dev[name] = self.cuda.mem_alloc(self.out_host[name].nbytes)

    def infer(self, tensors: list) -> dict:
        """Batch inference. Returns parsed results per frame."""
        B = len(tensors)
        md = self.max_detections

        if self.is_dynamic:
            self.context.set_input_shape(self.input_name, (B, 3, self.height, self.width))

        # copy batch to GPU
        batch = np.stack(tensors)
        np.copyto(self.inp_host[:batch.size], batch.ravel())
        self.cuda.memcpy_htod_async(self.inp_dev, self.inp_host, self.stream)

        self.context.set_tensor_address(self.input_name, int(self.inp_dev))
        for name in self.out_host:
            self.context.set_tensor_address(name, int(self.out_dev[name]))

        self.context.execute_async_v3(self.stream.handle)

        for name in self.out_host:
            self.cuda.memcpy_dtoh_async(self.out_host[name], self.out_dev[name], self.stream)
        self.stream.synchronize()

        # parse results per frame
        num_dets = self.out_host["num_detections"]
        boxes = self.out_host["boxes"]
        scores = self.out_host["scores"]
        classes = self.out_host["classes"]

        results = []
        for i in range(B):
            nd = int(num_dets[i])
            if nd > 0:
                b = boxes[i * md * 4:(i + 1) * md * 4][:nd * 4].reshape(nd, 4).copy()
                s = scores[i * md:(i + 1) * md][:nd].copy()
                c = classes[i * md:(i + 1) * md][:nd].astype(int).copy()
            else:
                b = np.zeros((0, 4), dtype=np.float32)
                s = np.zeros(0, dtype=np.float32)
                c = np.zeros(0, dtype=np.int32)
            results.append((b, s, c, nd))
        return results


# =============================================================================
# Frame container
# =============================================================================

class Frame:
    __slots__ = ("idx", "img", "tensor", "pad_info")

    def __init__(self, idx, img):
        self.idx = idx
        self.img = img
        self.tensor = None
        self.pad_info = None


# =============================================================================
# Video pipeline
# =============================================================================

def run_video(engine: TensorRTEngine, source, batch_size: int,
              conf: float, output: Optional[str], show: bool):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {W}x{H} @ {fps} FPS  batch={batch_size}\n")

    has_output = bool(output)
    writer_params = (output, fps, (W, H)) if output else None

    q_raw = Queue(maxsize=batch_size * 4)
    q_batch = Queue(maxsize=4)
    q_done = Queue(maxsize=batch_size * 4)

    frame_count = [0]
    infer_times = []
    infer_count = [0]

    # ---- reader ----
    def reader():
        while True:
            ret, img = cap.read()
            if not ret:
                break
            q_raw.put(Frame(frame_count[0], img))
            frame_count[0] += 1
        q_raw.put(None)

    # ---- preprocessor ----
    def preprocessor():
        buf = []
        while True:
            f = q_raw.get()
            if f is None:
                if buf:
                    q_batch.put(buf)
                q_batch.put(None)
                break
            f.tensor, f.pad_info = preprocess(f.img, engine.height, engine.width)
            buf.append(f)
            if len(buf) >= batch_size:
                q_batch.put(buf)
                buf = []

    # ---- inference ----
    def inference():
        while True:
            batch = q_batch.get()
            if batch is None:
                q_done.put(None)
                break
            t0 = time.perf_counter()
            results = engine.infer([f.tensor for f in batch])
            infer_times.append(time.perf_counter() - t0)
            infer_count[0] += 1
            for frame, (boxes, scores, classes, nd) in zip(batch, results):
                q_done.put((frame, boxes, scores, classes, nd))

    # ---- writer thread ----
    q_write = Queue(maxsize=batch_size * 2)

    def writer_thread(out_path, out_fps, out_size):
        w = make_writer(out_path, out_fps, out_size)
        while True:
            item = q_write.get()
            if item is None:
                break
            w.write(item)
        w.release()



    # ---- renderer ----
    fps_buf = []

    def renderer():
        while True:
            item = q_done.get()
            if item is None:
                break

            frame, boxes, scores, classes, nd = item

            # scale coordinates back
            if nd > 0:
                boxes = scale_boxes(boxes, frame.pad_info)

            # draw
            drawn = draw_detections(frame.img, boxes, scores, classes, conf)

            # FPS (1-second sliding window)
            now = time.perf_counter()
            fps_buf.append(now)
            while fps_buf and fps_buf[0] < now - 1.0:
                fps_buf.pop(0)
            cur_fps = len(fps_buf)

            avg_infer = np.mean(infer_times[-10:]) * 1000 if infer_times else 0

            drawn = draw_stats(drawn, cur_fps, nd, mode="LOCAL",
                               batch=batch_size, infer_ms=avg_infer)

            if has_output:
                q_write.put(drawn)
            if show:
                cv2.imshow("YOLO TensorRT", drawn)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

    # ---- start ----
    threads = [
        threading.Thread(target=reader, daemon=True),
        threading.Thread(target=preprocessor, daemon=True),
        threading.Thread(target=inference, daemon=True),
        threading.Thread(target=renderer, daemon=True),
    ]
    t_writer = None
    if has_output:
        t_writer = threading.Thread(target=writer_thread, args=writer_params)
        threads.append(t_writer)
    for t in threads:
        t.start()

    t0 = time.perf_counter()
    threads[3].join()  # wait for renderer (index 3)

    if t_writer:
        q_write.put(None)  # signal writer to finish
        t_writer.join()

    total = time.perf_counter() - t0
    processed = frame_count[0]

    print(f"\n{'='*50}")
    print("STATISTICS")
    print(f"{'='*50}")
    print(f"  Frames : {processed}")
    print(f"  Batches: {infer_count[0]}")
    print(f"  Time   : {total:.1f}s")
    print(f"  Pipeline FPS : {processed / total:.1f}")
    if infer_times:
        gpu_time = sum(infer_times)
        print(f"  GPU FPS      : {processed / gpu_time:.1f}")
        print(f"  Avg batch ms : {np.mean(infer_times)*1000:.1f}")

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# Image folder
# =============================================================================

def run_images(engine: TensorRTEngine, source_dir: str,
               batch_size: int, conf: float, output_dir: Optional[str]):
    src = Path(source_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in src.rglob("*") if f.suffix.lower() in exts)
    if not files:
        print(f"No images found in {source_dir}")
        return

    print(f"  {len(files)} images  batch={batch_size}\n")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_t = 0.0
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        frames, tensors = [], []
        for f in batch_files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            t, pad = preprocess(img, engine.height, engine.width)
            frames.append((img, pad))
            tensors.append(t)
        if not tensors:
            continue

        t0 = time.perf_counter()
        results = engine.infer(tensors)
        dt = time.perf_counter() - t0
        total_t += dt

        for j, ((img, pad), (boxes, scores, classes, nd)) in enumerate(zip(frames, results)):
            if nd > 0:
                boxes = scale_boxes(boxes, pad)
            drawn = draw_detections(img, boxes, scores, classes, conf)
            if output_dir:
                out = Path(output_dir) / f"{batch_files[j].stem}_det{batch_files[j].suffix}"
                cv2.imwrite(str(out), drawn)

        total_nd = sum(r[3] for r in results)
        print(f"  batch {i//batch_size+1}: {len(tensors)} img, {total_nd} det, {dt*1000:.1f}ms")

    print(f"\n  Total: {len(files)} images, {total_t:.2f}s, {len(files)/total_t:.1f} FPS")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="YOLO TensorRT — local inference")
    p.add_argument("--model", required=True, help="Path to .engine file")
    p.add_argument("--source", required=True, help="Video, camera (0,1...), or image folder")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--output", default=None, help="Output video file or image folder")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    engine = TensorRTEngine(args.model)
    batch = args.batch or engine.max_batch

    if not engine.is_dynamic and batch != engine.max_batch:
        print(f"Model batch={engine.max_batch}, requested {batch}. "
              f"Use --batch {engine.max_batch} or re-export with dynamic batch.")
        return

    if Path(args.source).is_dir():
        run_images(engine, args.source, batch, args.conf, args.output)
    else:
        run_video(engine, args.source, batch, args.conf, args.output, not args.no_show)


if __name__ == "__main__":
    main()
