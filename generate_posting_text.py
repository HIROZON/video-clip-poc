"""Generate per-platform posting text (title/description/hashtags/CTA) for each
clip candidate from analysis_result.json, via GPT-4o mini Structured Outputs.

One call is made per (clip, platform) pair, since each platform gets a
different character-length constraint. These are rough, commonly-cited
practical limits for a PoC, not each platform's hard technical maximum -
re-check current platform guidance before production use:
  - Instagram: title <=60 chars, description <=125 chars, <=10 hashtags, CTA <=30 chars
  - TikTok: title <=40 chars, description <=100 chars, <=5 hashtags, CTA <=20 chars
  - YouTube Shorts: title <=100 chars, description <=200 chars, <=15 hashtags, CTA <=30 chars

Usage:
    python generate_posting_text.py [--analysis analysis_result.json] [--out posting_texts.json]
                                     [--cost-log posting_cost.json]

Requires OPENAI_API_KEY to be set in the environment (optionally via a .env file).

Pricing note: gpt-4o-mini is billed at $0.15 / 1M input tokens and
$0.60 / 1M output tokens (confirmed via web search at the time this script was
written; re-check https://openai.com/api/pricing/ before production use).
"""
import argparse
import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gpt-4o-mini"
INPUT_RATE_USD_PER_1M = 0.15
OUTPUT_RATE_USD_PER_1M = 0.60

PLATFORM_CONSTRAINTS = {
    "instagram": {"title_max": 60, "description_max": 125, "hashtags_max": 10, "cta_max": 30},
    "tiktok": {"title_max": 40, "description_max": 100, "hashtags_max": 5, "cta_max": 20},
    "youtube_shorts": {"title_max": 100, "description_max": 200, "hashtags_max": 15, "cta_max": 30},
}


class PostingText(BaseModel):
    title: str = Field(description="Attention-grabbing title/hook for this clip")
    description: str = Field(description="Short description/caption body")
    hashtags: list[str] = Field(description="Relevant hashtags, without the leading #")
    cta: str = Field(description="Call-to-action text for this clip")


def build_prompt(clip: dict, platform: str, constraints: dict) -> str:
    return (
        f"以下の短尺クリップ候補向けに、{platform} 用の投稿文を作成してください。\n"
        f"クリップ区間: {clip['start_sec']:.2f}秒 - {clip['end_sec']:.2f}秒\n"
        f"選定理由: {clip['reason']}\n"
        f"要点の伝わりやすさスコア: {clip['clarity_score']}/5, CTAの自然さスコア: {clip['cta_score']}/5\n\n"
        f"文字数制約:\n"
        f"- title: 全角/半角問わず{constraints['title_max']}文字以内\n"
        f"- description: {constraints['description_max']}文字以内\n"
        f"- hashtags: 最大{constraints['hashtags_max']}個(#は付けない)\n"
        f"- cta: {constraints['cta_max']}文字以内\n"
    )


def generate_posting_text(analysis_path: str, out_path: str, cost_log_path: str) -> dict:
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    candidates = analysis["candidates"]

    client = OpenAI()
    clips_out = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    call_count = 0

    for clip in candidates:
        platforms_out = {}
        for platform, constraints in PLATFORM_CONSTRAINTS.items():
            prompt = build_prompt(clip, platform, constraints)
            completion = client.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "あなたはSNSショート動画の投稿文を作成するアシスタントです。文字数制約は厳守してください。"},
                    {"role": "user", "content": prompt},
                ],
                response_format=PostingText,
            )
            parsed = completion.choices[0].message.parsed
            usage = completion.usage

            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            call_count += 1

            platforms_out[platform] = parsed.model_dump()

        clips_out.append({
            "start_sec": clip["start_sec"],
            "end_sec": clip["end_sec"],
            "reason": clip["reason"],
            "platforms": platforms_out,
        })

    result = {"clips": clips_out}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    cost_usd = (total_prompt_tokens / 1_000_000) * INPUT_RATE_USD_PER_1M + (total_completion_tokens / 1_000_000) * OUTPUT_RATE_USD_PER_1M
    cost_log = {
        "stage": "generate_posting_text",
        "model": MODEL,
        "call_count": call_count,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "input_rate_usd_per_1m": INPUT_RATE_USD_PER_1M,
        "output_rate_usd_per_1m": OUTPUT_RATE_USD_PER_1M,
        "cost_usd": cost_usd,
    }
    with open(cost_log_path, "w", encoding="utf-8") as f:
        json.dump(cost_log, f, indent=2)

    print(f"[generate_posting_text] clips={len(candidates)} calls={call_count} (platforms={list(PLATFORM_CONSTRAINTS.keys())})")
    print(f"[generate_posting_text] prompt_tokens={total_prompt_tokens} completion_tokens={total_completion_tokens}")
    print(
        f"[generate_posting_text] rates: input=${INPUT_RATE_USD_PER_1M}/1M output=${OUTPUT_RATE_USD_PER_1M}/1M "
        f"(gpt-4o-mini; このレートは実行時点のものなので本番導入前に公式料金ページで再確認すること) "
        f"-> cost=${cost_usd:.6f}"
    )
    print(f"[generate_posting_text] wrote {out_path}, {cost_log_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate per-platform posting text via GPT-4o mini.")
    parser.add_argument("--analysis", default="analysis_result.json", help="Path to analysis_result.json")
    parser.add_argument("--out", default="posting_texts.json", help="Output posting texts JSON path")
    parser.add_argument("--cost-log", default="posting_cost.json", help="Output posting-text cost log JSON path")
    args = parser.parse_args()

    generate_posting_text(args.analysis, args.out, args.cost_log)


if __name__ == "__main__":
    main()
