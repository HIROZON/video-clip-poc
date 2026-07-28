"""Experimental: run MediaPipe Pose Landmarker on extracted frames.

Purpose: check how much Pose Landmarker (tracking shoulders/nose, i.e. the
upper body) can cover for frames where face detection fails - e.g. when the
subject has turned away or is looking off to the side, a face detector finds
nothing but shoulders are still visible and could anchor a crop.

This is a standalone experiment; it is not wired into smooth_and_crop.py.

Usage:
    python detect_pose.py [--frames frames] [--model models/pose_landmarker_lite.task] [--out pose.json]

Output JSON format (one entry per frame):
    {
        "frame": "frame_00000.jpg",
        "detected": true,
        "landmarks": {
            "nose":           {"x": 0.51, "y": 0.22, "visibility": 0.98},
            "left_shoulder":  {"x": 0.42, "y": 0.35, "visibility": 0.95},
            "right_shoulder": {"x": 0.60, "y": 0.36, "visibility": 0.93}
        }
    }
Landmark x/y are normalized to [0, 1] of the frame's width/height (MediaPipe's
standard NormalizedLandmark convention), not absolute pixel coordinates.
"""
import argparse
import glob
import json
import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Indices into PoseLandmarker's 33-point body model.
LANDMARKS_OF_INTEREST = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
}


def detect_poses(frames_dir: str, model_path: str, out_path: str) -> list[dict]:
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_poses=1,
    )

    results = []
    detected_count = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_path in frame_paths:
            frame_name = os.path.basename(frame_path)
            image_bgr = cv2.imread(frame_path)
            if image_bgr is None:
                results.append({"frame": frame_name, "detected": False})
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            pose_result = landmarker.detect(mp_image)

            if not pose_result.pose_landmarks:
                results.append({"frame": frame_name, "detected": False})
                continue

            landmarks = pose_result.pose_landmarks[0]
            landmark_out = {}
            for name, idx in LANDMARKS_OF_INTEREST.items():
                lm = landmarks[idx]
                landmark_out[name] = {
                    "x": lm.x,
                    "y": lm.y,
                    "visibility": lm.visibility,
                }

            results.append({"frame": frame_name, "detected": True, "landmarks": landmark_out})
            detected_count += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[detect_pose] frames={len(frame_paths)} detected={detected_count} -> {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect upper-body pose landmarks in extracted frames using MediaPipe.")
    parser.add_argument("--frames", default="frames", help="Directory containing extracted frames")
    parser.add_argument("--model", default="models/pose_landmarker_lite.task", help="Path to MediaPipe pose landmarker model")
    parser.add_argument("--out", default="pose.json", help="Output JSON path")
    args = parser.parse_args()

    detect_poses(args.frames, args.model, args.out)


if __name__ == "__main__":
    main()
