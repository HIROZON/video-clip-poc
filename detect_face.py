"""Run MediaPipe Face Detection on extracted frames and dump bounding boxes to JSON.

Usage:
    python detect_face.py [--frames frames] [--model models/blaze_face_short_range.tflite] [--out coordinates.json]

Output JSON format:
    [
        {"frame": "frame_00000.jpg", "detected": true, "x": 120, "y": 80, "width": 200, "height": 200},
        {"frame": "frame_00001.jpg", "detected": false},
        ...
    ]
Coordinates are absolute pixel values (top-left corner x, y, plus width/height).
"""
import argparse
import glob
import json
import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


def detect_faces(frames_dir: str, model_path: str, out_path: str) -> list[dict]:
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        min_detection_confidence=0.5,
    )

    results = []
    detected_count = 0
    with vision.FaceDetector.create_from_options(options) as detector:
        for frame_path in frame_paths:
            image_bgr = cv2.imread(frame_path)
            if image_bgr is None:
                results.append({"frame": os.path.basename(frame_path), "detected": False})
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)

            if detection_result.detections:
                bbox = detection_result.detections[0].bounding_box
                results.append({
                    "frame": os.path.basename(frame_path),
                    "detected": True,
                    "x": int(bbox.origin_x),
                    "y": int(bbox.origin_y),
                    "width": int(bbox.width),
                    "height": int(bbox.height),
                })
                detected_count += 1
            else:
                results.append({"frame": os.path.basename(frame_path), "detected": False})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[detect_face] frames={len(frame_paths)} detected={detected_count} -> {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect faces in extracted frames using MediaPipe.")
    parser.add_argument("--frames", default="frames", help="Directory containing extracted frames")
    parser.add_argument("--model", default="models/blaze_face_short_range.tflite", help="Path to MediaPipe face detector model")
    parser.add_argument("--out", default="coordinates.json", help="Output JSON path")
    args = parser.parse_args()

    detect_faces(args.frames, args.model, args.out)


if __name__ == "__main__":
    main()
