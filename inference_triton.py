"""Triton YOLO client — async gRPC.

Usage:
    python inference_triton.py --source video.mp4
    python inference_triton.py --source rtsp://192.168.1.100:554/stream --conf 0.3
"""

import argparse
import sys
import threading
import time
from functools import partial
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

from common import preprocess, scale_boxes, draw_detections, draw_stats, make_writer


class TritonClient:
    def __init__(self, url: str, model: str, h: int, w: int, batch_size: int = 1):
        self.url = url
        self.model = model
        self.h = h
        self.w = w
        self.batch_size = batch_size

        self.q_req = Queue(maxsize=256)
        self.q_res = Queue()

        self.captured = 0
        self.inferred = 0
        self.displayed = 0
        self.infer_times = []
        self.lock = threading.Lock()
        self.stop = threading.Event()

    # ---- preprocess ----
    def _preprocess(self, frame):
        tensor, pad_info = preprocess(frame, self.h, self.w)
        return {
            "frame": frame,
            "tensor": tensor,
            "pad_info": pad_info,
        }

    # ---- callback ----
    def _on_result(self, result, error, batch_reqs, send_time):
        if error:
            print(f"[Triton] Error: {error}")
            return
        if self.stop.is_set():
            return

        dt = time.perf_counter() - send_time
        num_dets = result.as_numpy("num_detections")
        boxes = result.as_numpy("boxes")
        scores = result.as_numpy("scores")
        classes = result.as_numpy("classes")

        for i, req in enumerate(batch_reqs):
            n = int(num_dets[i, 0])
            if n > 0:
                b = scale_boxes(boxes[i, :n].copy(), req["pad_info"])
                s = scores[i, :n].copy()
                c = classes[i, :n].copy().astype(int)
            else:
                b = np.zeros((0, 4), dtype=int)
                s = np.zeros(0)
                c = np.zeros(0, dtype=int)

            self.q_res.put({
                "frame": req["frame"],
                "boxes": b, "scores": s, "classes": c,
                "num_det": n, "infer_ms": dt * 1000,
            })

        with self.lock:
            self.inferred += len(batch_reqs)
            self.infer_times.append(dt)

    # ---- capture thread ----
    def _capture(self, video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        n = 0
        while not self.stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            self.q_req.put(self._preprocess(frame))
            n += 1
        cap.release()
        self.q_req.put(None)
        self.captured = n
        print(f"[Capture] {n} frames")

    # ---- infer thread ----
    def _infer(self):
        client = grpcclient.InferenceServerClient(
            url=self.url,
            channel_args=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
                ("grpc.keepalive_time_ms", 5000),
            ],
        )
        buf = []
        while not self.stop.is_set():
            try:
                req = self.q_req.get(timeout=0.1)
            except Empty:
                continue
            if req is None:
                if buf:
                    self._send(client, buf)
                break
            buf.append(req)
            if len(buf) >= self.batch_size:
                self._send(client, buf)
                buf = []

        # wait for pending async requests
        for _ in range(200):
            with self.lock:
                if self.inferred >= self.captured:
                    break
            time.sleep(0.05)
        self.q_res.put(None)
        print(f"[Infer] {self.inferred} frames")

    def _send(self, client, reqs):
        batch = np.stack([r["tensor"] for r in reqs])
        B = len(reqs)
        inp = grpcclient.InferInput("images", [B, 3, self.h, self.w], "FP32")
        inp.set_data_from_numpy(batch)
        outputs = [
            grpcclient.InferRequestedOutput("num_detections"),
            grpcclient.InferRequestedOutput("boxes"),
            grpcclient.InferRequestedOutput("scores"),
            grpcclient.InferRequestedOutput("classes"),
        ]
        client.async_infer(
            model_name=self.model,
            inputs=[inp],
            outputs=outputs,
            callback=partial(self._on_result, batch_reqs=reqs, send_time=time.perf_counter()),
        )

    # ---- writer thread ----
    def _writer(self, output, out_fps, out_size, q_write):
        w = make_writer(output, out_fps, out_size)
        while True:
            item = q_write.get()
            if item is None:
                break
            w.write(item)
        w.release()

    # ---- display thread ----
    def _display(self, conf, no_show, q_write=None):
        fps_buf = []
        last = None

        while not self.stop.is_set():
            try:
                item = self.q_res.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                break

            now = time.perf_counter()
            if last:
                fps_buf.append(now - last)
                if len(fps_buf) > 60:
                    fps_buf.pop(0)
            last = now
            cur_fps = len(fps_buf) / sum(fps_buf) if fps_buf else 0

            mask = item["scores"] >= conf if item["num_det"] > 0 else np.array([], dtype=bool)
            boxes = item["boxes"][mask] if np.any(mask) else np.zeros((0, 4), dtype=int)
            scores = item["scores"][mask] if np.any(mask) else np.zeros(0)
            classes = item["classes"][mask] if np.any(mask) else np.zeros(0, dtype=int)
            num_det = int(mask.sum()) if np.any(mask) else 0

            drawn = draw_detections(item["frame"], boxes, scores, classes, conf)
            drawn = draw_stats(drawn, cur_fps, num_det, mode="TRITON",
                               batch=self.batch_size, infer_ms=item["infer_ms"])

            if q_write:
                q_write.put(drawn)
            if not no_show:
                cv2.imshow("YOLO Triton", drawn)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop.set()
                    break

            self.displayed += 1

        cv2.destroyAllWindows()
        print(f"[Display] {self.displayed} frames")

    # ---- run ----
    def run(self, video_path: str, conf: float, no_show: bool, output: Optional[str] = None):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"  {W}x{H} @ {fps} FPS  total={total}")
        print(f"  batch={self.batch_size}  async gRPC")
        if output:
            print(f"  output: {output}")
        print()

        q_write = None
        t_writer = None
        if output:
            q_write = Queue(maxsize=32)
            t_writer = threading.Thread(target=self._writer, args=(output, fps, (W, H), q_write))

        t_cap = threading.Thread(target=self._capture, args=(video_path,), daemon=True)
        t_inf = threading.Thread(target=self._infer, daemon=True)
        t_dis = threading.Thread(target=self._display,
                                args=(conf, no_show, q_write), daemon=True)

        t0 = time.time()
        t_cap.start(); t_inf.start(); t_dis.start()
        if t_writer:
            t_writer.start()

        try:
            while t_dis.is_alive():
                t_dis.join(timeout=0.5)
        except KeyboardInterrupt:
            self.stop.set()

        if q_write:
            q_write.put(None)
            t_writer.join()

        self.stop.set()
        for t in (t_cap, t_inf, t_dis):
            t.join(timeout=3)

        wall = time.time() - t0
        print(f"\n{'='*50}")
        print(f"STATISTICS")
        print(f"{'='*50}")
        print(f"  Frames   : {total}")
        print(f"  Captured : {self.captured}")
        print(f"  Inferred : {self.inferred}")
        print(f"  Displayed: {self.displayed}")
        print(f"  Wall time: {wall:.1f}s")

        if self.infer_times:
            avg = np.mean(self.infer_times) / self.batch_size
            print(f"  Avg infer: {avg*1000:.1f}ms ({1/avg:.0f} FPS)")

        if total > 0 and wall > 0:
            speed = (total / fps) / wall
            print(f"  Speed: {speed:.2f}x realtime {'OK' if speed >= 1 else 'SLOW'}")


def main():
    p = argparse.ArgumentParser(description="YOLO Triton — async gRPC client")
    p.add_argument("--source", required=True, help="Video, camera (0,1...), RTSP stream, or image folder")
    p.add_argument("--url", default="localhost:8001")
    p.add_argument("--model", default="yolo_detector")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--batch", type=int, default=1, help="Client-side batching (1=disabled)")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--output", default=None, help="Output video file path")
    args = p.parse_args()

    print("\n" + "=" * 50)
    print("YOLO Triton — async gRPC")
    print("=" * 50)

    client = grpcclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        print("Triton not ready")
        return
    print("Triton ready\n")

    tc = TritonClient(
        url=args.url,
        model=args.model,
        h=args.height,
        w=args.width,
        batch_size=args.batch,
    )
    tc.run(args.source, args.conf, args.no_show, args.output)


if __name__ == "__main__":
    main()
