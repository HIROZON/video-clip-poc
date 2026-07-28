"""Aggregate the cost logs written by transcribe.py, analyze.py, and
generate_posting_text.py into a total cost per video.

Usage:
    python measure_total_cost.py [--asr-cost asr_cost.json] [--analyze-cost analyze_cost.json]
                                  [--posting-cost posting_cost.json] [--jpy-rate 150]
"""
import argparse
import json


def load_cost(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Aggregate ASR/LLM cost logs into a total cost per video.")
    parser.add_argument("--asr-cost", default="asr_cost.json", help="Path to the ASR cost log from transcribe.py")
    parser.add_argument("--analyze-cost", default="analyze_cost.json", help="Path to the analyze cost log from analyze.py")
    parser.add_argument("--posting-cost", default="posting_cost.json", help="Path to the posting-text cost log from generate_posting_text.py")
    parser.add_argument("--jpy-rate", type=float, default=150.0, help="USD->JPY conversion rate to use for the yen figure")
    args = parser.parse_args()

    asr = load_cost(args.asr_cost)
    analyze_cost = load_cost(args.analyze_cost)
    posting = load_cost(args.posting_cost)

    total_usd = asr["cost_usd"] + analyze_cost["cost_usd"] + posting["cost_usd"]
    total_jpy = total_usd * args.jpy_rate

    print("[measure_total_cost] --- breakdown (USD) ---")
    print(f"[measure_total_cost] ASR (transcribe.py)              : ${asr['cost_usd']:.6f}")
    print(f"[measure_total_cost] LLM analysis (analyze.py)         : ${analyze_cost['cost_usd']:.6f}")
    print(f"[measure_total_cost] posting text (generate_posting_text.py): ${posting['cost_usd']:.6f}")
    print(f"[measure_total_cost] TOTAL                             : ${total_usd:.6f}")
    print(f"[measure_total_cost] TOTAL (JPY @ {args.jpy_rate:.0f}円/$)          : {total_jpy:.2f}円")


if __name__ == "__main__":
    main()
