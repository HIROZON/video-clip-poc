"""Extract the audio track from a video as an mp3 file, via ffmpeg.

Usage:
    python extract_audio.py [--video samples/sample_16x9.mp4] [--out audio/sample_16x9.mp3]
"""
import argparse
import os
import subprocess


def extract_audio(video_path: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}")

    print(f"[extract_audio] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract audio from a video as mp3.")
    parser.add_argument("--video", default="samples/sample_16x9.mp4", help="Path to the source video file")
    parser.add_argument("--out", default="audio/sample_16x9.mp3", help="Output mp3 path")
    args = parser.parse_args()

    extract_audio(args.video, args.out)


if __name__ == "__main__":
    main()
