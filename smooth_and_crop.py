"""Smooth face-center coordinates and produce a 9:16 vertical crop of the source video.

Steps:
  1. Load per-frame face bounding boxes from coordinates.json (produced by detect_face.py).
  2. For frames where detection failed, fall back to the last known-good center.
     (This naturally also covers the "3+ consecutive failures" case called out in the
     spec: the held center simply keeps being reused until detection recovers.)
  3. Smooth the resulting per-frame centers with a trailing moving average over the
     last 10 samples.
  4. Compute a single 9:16 crop window size from the source resolution, and a
     per-timestamp crop position from the smoothed centers.
  5. Drive ffmpeg's crop filter with a piecewise time expression to produce
     output_vertical.mp4.

Usage:
    python smooth_and_crop.py <input_video> [--coords coordinates.json] [--interval 0.5] [--out output_vertical.mp4]
"""
import argparse
import json
import subprocess

import cv2

WINDOW = 10
FALLBACK_STREAK_THRESHOLD = 3


def load_smoothed_centers(coords_path: str, frame_w: int, frame_h: int):
    with open(coords_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    last_known = None
    consecutive_failures = 0
    raw_centers = []
    fallback_frame_count = 0

    for entry in entries:
        if entry.get("detected"):
            cx = entry["x"] + entry["width"] / 2
            cy = entry["y"] + entry["height"] / 2
            last_known = (cx, cy)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if last_known is None:
                # No detection has ever succeeded yet; default to the frame center.
                last_known = (frame_w / 2, frame_h / 2)
            if consecutive_failures >= FALLBACK_STREAK_THRESHOLD:
                fallback_frame_count += 1
            cx, cy = last_known
        raw_centers.append((cx, cy))

    smoothed = []
    for i in range(len(raw_centers)):
        window = raw_centers[max(0, i - WINDOW + 1): i + 1]
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        smoothed.append((avg_x, avg_y))

    print(f"[smooth_and_crop] frames={len(entries)} fallback_frames(streak>={FALLBACK_STREAK_THRESHOLD})={fallback_frame_count}")
    return smoothed


def compute_crop_size(frame_w: int, frame_h: int):
    target_ratio = 9 / 16  # width / height
    if frame_w / frame_h > target_ratio:
        crop_h = frame_h
        crop_w = round(crop_h * target_ratio)
    else:
        crop_w = frame_w
        crop_h = round(crop_w / target_ratio)

    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    return crop_w, crop_h


def build_position_expr(values, timestamps):
    """Build a piecewise ffmpeg time expression: value holds until the next timestamp."""
    expr = f"{values[-1]:.2f}"
    for t, v in reversed(list(zip(timestamps[1:], values[:-1]))):
        expr = f"if(lt(t,{t:.3f}),{v:.2f},{expr})"
    return expr


def run(video_path: str, coords_path: str, interval: float, out_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    smoothed_centers = load_smoothed_centers(coords_path, frame_w, frame_h)
    crop_w, crop_h = compute_crop_size(frame_w, frame_h)

    timestamps = [i * interval for i in range(len(smoothed_centers))]
    xs, ys = [], []
    for cx, cy in smoothed_centers:
        x = cx - crop_w / 2
        y = cy - crop_h / 2
        x = min(max(x, 0), frame_w - crop_w)
        y = min(max(y, 0), frame_h - crop_h)
        xs.append(x)
        ys.append(y)

    x_expr = build_position_expr(xs, timestamps)
    y_expr = build_position_expr(ys, timestamps)

    vf = f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_path,
    ]
    print(f"[smooth_and_crop] source={frame_w}x{frame_h} crop={crop_w}x{crop_h} samples={len(smoothed_centers)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}")

    print(f"[smooth_and_crop] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Smooth face coordinates and crop video to 9:16.")
    parser.add_argument("video", help="Path to the source video file")
    parser.add_argument("--coords", default="coordinates.json", help="Path to coordinates.json")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between coordinate samples (must match extract_frames.py)")
    parser.add_argument("--out", default="output_vertical.mp4", help="Output video path")
    args = parser.parse_args()

    run(args.video, args.coords, args.interval, args.out)


if __name__ == "__main__":
    main()
