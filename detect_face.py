"""Run MediaPipe Face Detection on extracted frames and dump bounding boxes to JSON.

Usage:
    python detect_face.py [--frames frames] [--model models/blaze_face_full_range.tflite]
                           [--out coordinates.json] [--threshold 0.5]

Confidence handling
--------------------
MediaPipe's FaceDetector task applies its `min_detection_confidence` threshold
*inside* the graph: candidates scoring below it are discarded before the
result ever reaches Python, so a plain call at the real threshold can't tell
"a face-like candidate existed but scored too low" apart from "no candidate
at all". To recover that distinction, the detector here is created with a
near-zero internal threshold (so essentially all raw candidates survive) and
the real threshold (--threshold, default 0.5 - MediaPipe's own default) is
then applied in Python against each frame's best candidate score.

Output JSON format (one entry per frame):
    {
        "frame": "frame_00000.jpg",
        "attempted": true,
        "candidate_count": 2,
        "best_score": 0.42,
        "detected": false,
        "reason": "below_threshold"   # or "no_candidates"; omitted when detected
    }
    {
        "frame": "frame_00001.jpg",
        "attempted": true,
        "candidate_count": 1,
        "best_score": 0.91,
        "detected": true,
        "x": 120, "y": 80, "width": 200, "height": 200, "score": 0.91
    }
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

DEFAULT_THRESHOLD = 0.5  # MediaPipe FaceDetectorOptions.min_detection_confidence default
RAW_CANDIDATE_THRESHOLD = 0.01  # near-zero, so the graph itself discards almost nothing


def detect_faces(frames_dir: str, model_path: str, out_path: str, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    print(f"[detect_face] min_detection_confidence threshold = {threshold} (MediaPipe default is 0.5)")

    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        min_detection_confidence=RAW_CANDIDATE_THRESHOLD,
    )

    results = []
    detected_count = 0
    below_threshold_count = 0
    no_candidate_count = 0

    with vision.FaceDetector.create_from_options(options) as detector:
        for frame_path in frame_paths:
            frame_name = os.path.basename(frame_path)
            image_bgr = cv2.imread(frame_path)
            if image_bgr is None:
                results.append({"frame": frame_name, "attempted": False, "detected": False, "reason": "read_error"})
                no_candidate_count += 1
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)

            candidates = detection_result.detections
            candidate_count = len(candidates)

            if candidate_count == 0:
                results.append({
                    "frame": frame_name,
                    "attempted": True,
                    "candidate_count": 0,
                    "best_score": None,
                    "detected": False,
                    "reason": "no_candidates",
                })
                no_candidate_count += 1
                continue

            best = max(candidates, key=lambda d: d.categories[0].score)
            best_score = best.categories[0].score

            if best_score >= threshold:
                bbox = best.bounding_box
                results.append({
                    "frame": frame_name,
                    "attempted": True,
                    "candidate_count": candidate_count,
                    "best_score": best_score,
                    "detected": True,
                    "x": int(bbox.origin_x),
                    "y": int(bbox.origin_y),
                    "width": int(bbox.width),
                    "height": int(bbox.height),
                    "score": best_score,
                })
                detected_count += 1
            else:
                results.append({
                    "frame": frame_name,
                    "attempted": True,
                    "candidate_count": candidate_count,
                    "best_score": best_score,
                    "detected": False,
                    "reason": "below_threshold",
                })
                below_threshold_count += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"[detect_face] frames={len(frame_paths)} detected={detected_count} "
        f"below_threshold={below_threshold_count} no_candidates={no_candidate_count} -> {out_path}"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect faces in extracted frames using MediaPipe.")
    parser.add_argument("--frames", default="frames", help="Directory containing extracted frames")
    parser.add_argument("--model", default="models/blaze_face_full_range.tflite", help="Path to MediaPipe face detector model")
    parser.add_argument("--out", default="coordinates.json", help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum confidence score to accept a detection")
    args = parser.parse_args()

    detect_faces(args.frames, args.model, args.out, args.threshold)


if __name__ == "__main__":
    main()
