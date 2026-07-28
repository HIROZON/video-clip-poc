"""Smooth face/pose-center coordinates and produce a 9:16 vertical crop of the source video.

Steps:
  1. For each frame, resolve a center coordinate with a 3-tier priority:
       1) face detection (coordinates.json from detect_face.py) if it succeeded
       2) otherwise, Pose Landmarker (pose.json from detect_pose.py): the
          midpoint of both shoulders if both are visible enough, else the
          nose landmark, if either is visible enough
       3) otherwise, hold the last known-good center (the original fallback)
     A per-frame log of which tier was used is printed as a summary.
  2. Smooth the resulting per-frame centers with a trailing moving average over the
     last 10 samples.
  3. Compute a single 9:16 crop window size from the source resolution, and a
     per-timestamp crop position from the smoothed centers.
  4. Drive ffmpeg's crop filter via the sendcmd filter (one x/y update per
     timestamp) to produce output_vertical.mp4. sendcmd is used instead of a
     single nested if(lt(t,...),...) expression because that expression's
     nesting depth grows with the number of samples and exceeds ffmpeg's
     expression parser limit on anything longer than a few seconds of video.

Usage:
    python smooth_and_crop.py <input_video> [--coords coordinates.json] [--pose pose.json]
                               [--interval 0.5] [--out output_vertical.mp4]
"""
import argparse
import json
import os
import subprocess
import tempfile

import cv2

WINDOW = 10
POSE_VISIBILITY_THRESHOLD = 0.5


def load_smoothed_centers(coords_path: str, pose_path: str, frame_w: int, frame_h: int):
    with open(coords_path, "r", encoding="utf-8") as f:
        face_entries = json.load(f)

    pose_by_frame = {}
    if pose_path and os.path.exists(pose_path):
        with open(pose_path, "r", encoding="utf-8") as f:
            pose_by_frame = {e["frame"]: e for e in json.load(f)}

    last_known = None
    raw_centers = []
    source_counts = {"face": 0, "pose": 0, "fallback": 0}

    for entry in face_entries:
        center = None
        source = None

        if entry.get("detected"):
            cx = entry["x"] + entry["width"] / 2
            cy = entry["y"] + entry["height"] / 2
            center = (cx, cy)
            source = "face"
        else:
            pose_entry = pose_by_frame.get(entry["frame"])
            if pose_entry and pose_entry.get("detected"):
                lm = pose_entry["landmarks"]
                left_shoulder, right_shoulder, nose = lm["left_shoulder"], lm["right_shoulder"], lm["nose"]
                if left_shoulder["visibility"] >= POSE_VISIBILITY_THRESHOLD and right_shoulder["visibility"] >= POSE_VISIBILITY_THRESHOLD:
                    cx = (left_shoulder["x"] + right_shoulder["x"]) / 2 * frame_w
                    cy = (left_shoulder["y"] + right_shoulder["y"]) / 2 * frame_h
                    center = (cx, cy)
                    source = "pose"
                elif nose["visibility"] >= POSE_VISIBILITY_THRESHOLD:
                    center = (nose["x"] * frame_w, nose["y"] * frame_h)
                    source = "pose"

        if center is None:
            # Both face and pose failed for this frame; hold the last known-good center.
            center = last_known if last_known is not None else (frame_w / 2, frame_h / 2)
            source = "fallback"
        else:
            last_known = center

        raw_centers.append(center)
        source_counts[source] += 1

    smoothed = []
    for i in range(len(raw_centers)):
        window = raw_centers[max(0, i - WINDOW + 1): i + 1]
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        smoothed.append((avg_x, avg_y))

    print(
        f"[smooth_and_crop] frames={len(face_entries)} "
        f"center_source: face={source_counts['face']} pose={source_counts['pose']} fallback={source_counts['fallback']}"
    )
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


def write_sendcmd_file(xs, ys, timestamps, path: str) -> None:
    """Write an ffmpeg sendcmd script: one crop x/y update per sample timestamp."""
    with open(path, "w", encoding="ascii") as f:
        for t, x, y in zip(timestamps, xs, ys):
            f.write(f"{t:.3f} crop x {x:.2f}, crop y {y:.2f};\n")


def ffmpeg_escape_path(path: str) -> str:
    """Escape a filesystem path for use as an ffmpeg filter option value."""
    return path.replace("\\", "/").replace(":", "\\:")


def run(video_path: str, coords_path: str, interval: float, out_path: str, pose_path: str = "pose.json") -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    smoothed_centers = load_smoothed_centers(coords_path, pose_path, frame_w, frame_h)
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

    fd, commands_path = tempfile.mkstemp(suffix=".txt", prefix="crop_cmds_")
    os.close(fd)
    try:
        write_sendcmd_file(xs, ys, timestamps, commands_path)
        escaped_commands_path = ffmpeg_escape_path(commands_path)

        vf = f"sendcmd=f='{escaped_commands_path}',crop=w={crop_w}:h={crop_h}:x={xs[0]:.2f}:y={ys[0]:.2f}"

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
    finally:
        os.remove(commands_path)

    print(f"[smooth_and_crop] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Smooth face coordinates and crop video to 9:16.")
    parser.add_argument("video", help="Path to the source video file")
    parser.add_argument("--coords", default="coordinates.json", help="Path to coordinates.json")
    parser.add_argument("--pose", default="pose.json", help="Path to pose.json (produced by detect_pose.py); used when face detection fails")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between coordinate samples (must match extract_frames.py)")
    parser.add_argument("--out", default="output_vertical.mp4", help="Output video path")
    args = parser.parse_args()

    run(args.video, args.coords, args.interval, args.out, args.pose)


if __name__ == "__main__":
    main()
