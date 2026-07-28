"""Convert transcript.json (Whisper verbose_json segments) into a standard SRT file.

Usage:
    python generate_srt.py [--transcript transcript.json] [--out subtitles.srt]
"""
import argparse
import json


def format_srt_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def generate_srt(transcript_path: str, out_path: str) -> str:
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    lines = []
    for i, seg in enumerate(transcript["segments"], start=1):
        start_ts = format_srt_timestamp(seg["start"])
        end_ts = format_srt_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    srt_content = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"[generate_srt] segments={len(transcript['segments'])} -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert transcript.json into an SRT subtitle file.")
    parser.add_argument("--transcript", default="transcript.json", help="Path to transcript.json")
    parser.add_argument("--out", default="subtitles.srt", help="Output SRT path")
    args = parser.parse_args()

    generate_srt(args.transcript, args.out)


if __name__ == "__main__":
    main()
