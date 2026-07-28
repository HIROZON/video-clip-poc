"""Burn subtitles.srt onto a video via ffmpeg's subtitles (libass) filter.

Usage:
    python burn_subtitles.py [--video output_vertical.mp4] [--srt subtitles.srt] [--out output_with_subs.mp4]

Japanese font handling
-----------------------
libass (used by ffmpeg's subtitles filter) needs a font that actually covers
Japanese glyphs, and this Windows box has no fontconfig config file (seen
earlier as "Fontconfig error: Cannot load default config file" from
drawtext), so we can't rely on family-name lookup through fontconfig. Instead
we point libass directly at the Windows Fonts folder via the filter's
`fontsdir` option and select a known-installed Japanese font by its embedded
family name (checked with fontTools ahead of time) via `force_style`.

If none of the candidate fonts are found on this machine, the script exits
with an error explaining that a Japanese-capable font must be installed
separately - it does not silently fall back to a font that would render
Japanese text as tofu boxes.
"""
import argparse
import os
import subprocess
import time

WINDOWS_FONTS_DIR = r"C:\Windows\Fonts"

# (font family name as embedded in the font file, filename to check for) - checked in priority order.
CANDIDATE_JAPANESE_FONTS = [
    ("MS Gothic", "msgothic.ttc"),
    ("Yu Gothic", "YuGothR.ttc"),
    ("Meiryo", "meiryo.ttc"),
    ("Noto Sans JP", "NotoSansJP-VF.ttf"),
]


def find_japanese_font(fonts_dir: str = WINDOWS_FONTS_DIR):
    for family_name, filename in CANDIDATE_JAPANESE_FONTS:
        if os.path.exists(os.path.join(fonts_dir, filename)):
            return family_name, filename
    return None, None


def ffmpeg_escape_path(path: str) -> str:
    """Escape a filesystem path for use as an ffmpeg filter option value."""
    return path.replace("\\", "/").replace(":", "\\:")


def burn_subtitles(video_path: str, srt_path: str, out_path: str) -> dict:
    family_name, filename = find_japanese_font()
    if family_name is None:
        raise RuntimeError(
            "日本語対応フォントが見つかりませんでした（"
            + ", ".join(f for _, f in CANDIDATE_JAPANESE_FONTS)
            + f" を {WINDOWS_FONTS_DIR} で探索）。"
            "このマシンには日本語字幕を正しく描画できるフォントが無いため、"
            "字幕を文字化け（豆腐/tofu表示）させないためにも処理を中断します。"
            "別途、日本語対応フォント（Noto Sans JP、Yu Gothic、MS Gothic 等）を"
            "インストールしてから再実行してください。"
        )

    srt_escaped = ffmpeg_escape_path(os.path.abspath(srt_path))
    fontsdir_escaped = ffmpeg_escape_path(WINDOWS_FONTS_DIR)

    vf = (
        f"subtitles=filename='{srt_escaped}':fontsdir='{fontsdir_escaped}':"
        f"force_style='FontName={family_name},Fontsize=20,PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=60'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_path,
    ]

    print(f"[burn_subtitles] using font: {family_name} ({filename}) from {WINDOWS_FONTS_DIR}")
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_sec = time.perf_counter() - start

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}")

    print(f"[burn_subtitles] wrote {out_path}")
    print(f"[burn_subtitles] elapsed={elapsed_sec:.3f}s")
    return {"font_family": family_name, "font_file": filename, "elapsed_sec": elapsed_sec, "out_path": out_path}


def main():
    parser = argparse.ArgumentParser(description="Burn SRT subtitles onto a video with ffmpeg's subtitles filter.")
    parser.add_argument("--video", default="output_vertical.mp4", help="Path to the source video file")
    parser.add_argument("--srt", default="subtitles.srt", help="Path to the SRT subtitle file")
    parser.add_argument("--out", default="output_with_subs.mp4", help="Output video path")
    args = parser.parse_args()

    burn_subtitles(args.video, args.srt, args.out)


if __name__ == "__main__":
    main()
