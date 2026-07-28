"""Run the full extract -> detect -> crop pipeline once and report timing.

Prints total wall-clock time and the normalized processing cost per 1 minute
of source video (seconds of processing per minute of input).

Usage:
    python measure.py <input_video> [--interval 0.5] [--frames frames]
                       [--model models/blaze_face_full_range.tflite]
                       [--coords coordinates.json] [--out output_vertical.mp4]
"""
import argparse
import time

import cv2

import extract_frames
import detect_face
import smooth_and_crop


def main():
    parser = argparse.ArgumentParser(description="Measure end-to-end pipeline processing time.")
    parser.add_argument("video", help="Path to the source video file")
    parser.add_argument("--interval", type=float, default=0.5, help="Frame sampling interval in seconds")
    parser.add_argument("--frames", default="frames", help="Directory for extracted frames")
    parser.add_argument("--model", default="models/blaze_face_full_range.tflite", help="Path to MediaPipe face detector model")
    parser.add_argument("--coords", default="coordinates.json", help="Output path for detected coordinates")
    parser.add_argument("--out", default="output_vertical.mp4", help="Output cropped video path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    source_duration_sec = frame_count / fps if fps else 0.0

    start = time.perf_counter()

    t0 = time.perf_counter()
    extract_frames.extract_frames(args.video, args.frames, args.interval)
    t1 = time.perf_counter()

    detect_face.detect_faces(args.frames, args.model, args.coords)
    t2 = time.perf_counter()

    smooth_and_crop.run(args.video, args.coords, args.interval, args.out)
    t3 = time.perf_counter()

    total_sec = time.perf_counter() - start

    print("[measure] --- stage timing (sec) ---")
    print(f"[measure] extract_frames : {t1 - t0:.3f}")
    print(f"[measure] detect_face    : {t2 - t1:.3f}")
    print(f"[measure] smooth_and_crop: {t3 - t2:.3f}")
    print(f"[measure] total          : {total_sec:.3f}")
    print(f"[measure] source video duration: {source_duration_sec:.3f} sec")

    if source_duration_sec > 0:
        sec_per_minute = total_sec / (source_duration_sec / 60.0)
        print(f"[measure] processing time per 1 min of source video: {sec_per_minute:.3f} sec/min")
    else:
        print("[measure] source video duration is 0; cannot compute per-minute rate")


if __name__ == "__main__":
    main()
