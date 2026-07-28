"""Extract frames from a video at a fixed time interval.

Usage:
    python extract_frames.py <input_video> [--interval 0.5] [--out frames]
"""
import argparse
import glob
import os

import cv2


def extract_frames(video_path: str, out_dir: str, interval_sec: float) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)

    # Remove frames left over from a previous run so a shorter video can't end up
    # with stale, higher-numbered frames from an earlier, longer one mixed in.
    for stale_frame in glob.glob(os.path.join(out_dir, "frame_*.jpg")):
        os.remove(stale_frame)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, round(fps * interval_sec))

    saved_paths = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            out_path = os.path.join(out_dir, f"frame_{saved_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"[extract_frames] fps={fps:.2f} step={frame_step} frames_saved={len(saved_paths)} -> {out_dir}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video at a fixed interval.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval between extracted frames in seconds")
    parser.add_argument("--out", default="frames", help="Output directory for extracted frames")
    args = parser.parse_args()

    extract_frames(args.video, args.out, args.interval)


if __name__ == "__main__":
    main()
