"""Transcribe an audio file with OpenAI's Whisper API (whisper-1), with timestamps.

Usage:
    python transcribe.py [--audio audio/sample_16x9.mp3] [--out transcript.json] [--cost-log asr_cost.json]

Requires OPENAI_API_KEY to be set in the environment (optionally via a .env file).
"""
import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RATE_USD_PER_MINUTE = 0.006  # whisper-1 pricing, as specified for this project


def transcribe(audio_path: str, out_path: str, cost_log_path: str) -> dict:
    client = OpenAI()

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    duration_sec = result.duration
    segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in (result.segments or [])
    ]
    transcript = {
        "duration_sec": duration_sec,
        "language": result.language,
        "text": result.text,
        "segments": segments,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    duration_min = duration_sec / 60.0
    cost_usd = duration_min * RATE_USD_PER_MINUTE
    cost_log = {
        "stage": "asr",
        "model": "whisper-1",
        "audio_duration_sec": duration_sec,
        "audio_duration_min": duration_min,
        "rate_usd_per_minute": RATE_USD_PER_MINUTE,
        "cost_usd": cost_usd,
    }
    with open(cost_log_path, "w", encoding="utf-8") as f:
        json.dump(cost_log, f, indent=2)

    print(f"[transcribe] segments={len(segments)} duration={duration_sec:.2f}s ({duration_min:.4f} min)")
    print(f"[transcribe] rate=${RATE_USD_PER_MINUTE}/min -> cost=${cost_usd:.6f} -> {out_path}, {cost_log_path}")
    return transcript


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with OpenAI Whisper API.")
    parser.add_argument("--audio", default="audio/sample_16x9.mp3", help="Path to the input audio file")
    parser.add_argument("--out", default="transcript.json", help="Output transcript JSON path")
    parser.add_argument("--cost-log", default="asr_cost.json", help="Output ASR cost log JSON path")
    args = parser.parse_args()

    transcribe(args.audio, args.out, args.cost_log)


if __name__ == "__main__":
    main()
