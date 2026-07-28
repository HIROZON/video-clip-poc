"""Analyze a transcript and propose short-form clip candidates using GPT-4o mini
Structured Outputs.

For each candidate clip, the model returns a start/end time (grounded in the
transcript's own segment timestamps), a one-line reason it works as a
short-form clip, and two 1-5 scores for a seminar/webinar use case: how well
it conveys its key point on its own, and how natural its call-to-action feels.

Usage:
    python analyze.py [--transcript transcript.json] [--out analysis_result.json] [--cost-log analyze_cost.json]

Requires OPENAI_API_KEY to be set in the environment (optionally via a .env file).

Pricing note: gpt-4o-mini is billed at $0.15 / 1M input tokens and
$0.60 / 1M output tokens (confirmed via web search at the time this script was
written). This rate can change - re-check the official OpenAI pricing page
(https://openai.com/api/pricing/) before relying on it for production cost
estimates.
"""
import argparse
import json
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gpt-4o-mini"
INPUT_RATE_USD_PER_1M = 0.15
OUTPUT_RATE_USD_PER_1M = 0.60

SYSTEM_PROMPT = """あなたはセミナー/ウェビナー動画から短尺クリップ(ショート動画)候補を選定する
アシスタントです。文字起こし(タイムスタンプ付き)を読み、切り抜き候補を3〜10個提案してください。

各候補について:
- start_sec, end_sec: 文字起こしのタイムスタンプに基づく開始・終了時刻(秒)。実在するセグメントの
  区切りに沿わせること。
- reason: なぜこの区間が短尺クリップ候補として有効か(例: 冒頭5秒で興味を引く、CTAが自然、等)。
- clarity_score: このクリップ単体で要点が伝わるか、1(伝わらない)〜5(非常によく伝わる)の5段階評価。
- cta_score: このクリップ内のCTA(視聴者への行動喚起)が自然か、1(不自然/CTAなし)〜5(非常に自然)の
  5段階評価。CTAが存在しない場合は低めのスコアにすること。
"""


class ClipCandidate(BaseModel):
    start_sec: float = Field(description="Clip start time in seconds, aligned to a transcript segment boundary")
    end_sec: float = Field(description="Clip end time in seconds, aligned to a transcript segment boundary")
    reason: str = Field(description="Why this clip works as a short-form candidate")
    clarity_score: int = Field(description="1-5: how well the clip conveys its key point on its own")
    cta_score: int = Field(description="1-5: how natural the call-to-action feels in this clip")


class ClipAnalysis(BaseModel):
    candidates: List[ClipCandidate]


def build_user_prompt(transcript: dict) -> str:
    lines = [f"動画長: {transcript['duration_sec']:.2f}秒", "", "文字起こし(タイムスタンプ付き):"]
    for seg in transcript["segments"]:
        lines.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text'].strip()}")
    return "\n".join(lines)


def analyze(transcript_path: str, out_path: str, cost_log_path: str) -> dict:
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    user_prompt = build_user_prompt(transcript)

    client = OpenAI()
    completion = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ClipAnalysis,
    )

    parsed = completion.choices[0].message.parsed
    usage = completion.usage

    result = parsed.model_dump()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    cost_usd = (usage.prompt_tokens / 1_000_000) * INPUT_RATE_USD_PER_1M + (usage.completion_tokens / 1_000_000) * OUTPUT_RATE_USD_PER_1M
    cost_log = {
        "stage": "analyze",
        "model": MODEL,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "input_rate_usd_per_1m": INPUT_RATE_USD_PER_1M,
        "output_rate_usd_per_1m": OUTPUT_RATE_USD_PER_1M,
        "cost_usd": cost_usd,
        "user_prompt_char_length": len(user_prompt),
    }
    with open(cost_log_path, "w", encoding="utf-8") as f:
        json.dump(cost_log, f, indent=2)

    print(f"[analyze] candidates={len(parsed.candidates)}")
    print(f"[analyze] prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens} total_tokens={usage.total_tokens}")
    print(f"[analyze] user_prompt_char_length={len(user_prompt)}")
    print(
        f"[analyze] rates: input=${INPUT_RATE_USD_PER_1M}/1M output=${OUTPUT_RATE_USD_PER_1M}/1M "
        f"(gpt-4o-mini; このレートは実行時点のものなので本番導入前に公式料金ページで再確認すること) "
        f"-> cost=${cost_usd:.6f}"
    )
    print(f"[analyze] wrote {out_path}, {cost_log_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze transcript and propose clip candidates via GPT-4o mini.")
    parser.add_argument("--transcript", default="transcript.json", help="Path to transcript.json")
    parser.add_argument("--out", default="analysis_result.json", help="Output analysis JSON path")
    parser.add_argument("--cost-log", default="analyze_cost.json", help="Output analyze cost log JSON path")
    args = parser.parse_args()

    analyze(args.transcript, args.out, args.cost_log)


if __name__ == "__main__":
    main()
