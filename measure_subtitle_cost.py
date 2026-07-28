"""Convert burn_subtitles.py's processing time into a Railway-hosting cost
estimate, and combine it with the earlier face-detection/reframe cost into a
total per-minute video-processing cost.

Railway pricing assumed (as given for this project):
    CPU: $0.00000772 / vCPU-second
    RAM: $0.00000386 / GB-second
    Instance sized at 1 vCPU / 1 GB RAM, fully utilized for the whole run.

Usage:
    python measure_subtitle_cost.py --elapsed-sec <burn_subtitles elapsed> --video-duration-sec <source duration>
                                     [--reframe-sec-per-min 24.2] [--jpy-rate 150]
"""
import argparse

CPU_RATE_USD_PER_VCPU_SEC = 0.00000772
RAM_RATE_USD_PER_GB_SEC = 0.00000386
VCPUS = 1
RAM_GB = 1


def railway_cost_usd(elapsed_sec: float) -> float:
    cpu_cost = elapsed_sec * VCPUS * CPU_RATE_USD_PER_VCPU_SEC
    ram_cost = elapsed_sec * RAM_GB * RAM_RATE_USD_PER_GB_SEC
    return cpu_cost + ram_cost


def main():
    parser = argparse.ArgumentParser(description="Estimate Railway hosting cost of subtitle burn-in and total video-processing cost.")
    parser.add_argument("--elapsed-sec", type=float, required=True, help="burn_subtitles.py elapsed time in seconds")
    parser.add_argument("--video-duration-sec", type=float, required=True, help="Source video duration in seconds, to normalize to a per-minute rate")
    parser.add_argument("--reframe-sec-per-min", type=float, default=24.2, help="Previously measured face-detection/reframe processing seconds per 1 minute of source video")
    parser.add_argument("--jpy-rate", type=float, default=150.0, help="USD->JPY conversion rate")
    args = parser.parse_args()

    video_duration_min = args.video_duration_sec / 60.0

    subtitle_cost_usd = railway_cost_usd(args.elapsed_sec)
    subtitle_sec_per_min = args.elapsed_sec / video_duration_min
    subtitle_cost_usd_per_min = railway_cost_usd(subtitle_sec_per_min)

    reframe_cost_usd_per_min = railway_cost_usd(args.reframe_sec_per_min)

    total_sec_per_min = subtitle_sec_per_min + args.reframe_sec_per_min
    total_cost_usd_per_min = subtitle_cost_usd_per_min + reframe_cost_usd_per_min

    print("[measure_subtitle_cost] --- subtitle burn-in ---")
    print(f"[measure_subtitle_cost] elapsed={args.elapsed_sec:.3f}s for {args.video_duration_sec:.2f}s source video ({video_duration_min:.4f} min)")
    print(f"[measure_subtitle_cost] normalized: {subtitle_sec_per_min:.3f} sec / 1 min of source video")
    print(f"[measure_subtitle_cost] Railway cost (this run): ${subtitle_cost_usd:.8f} ({subtitle_cost_usd * args.jpy_rate:.5f}円)")
    print(f"[measure_subtitle_cost] Railway cost per 1 min of source video: ${subtitle_cost_usd_per_min:.8f} ({subtitle_cost_usd_per_min * args.jpy_rate:.5f}円)")

    print("[measure_subtitle_cost] --- combined video-processing cost per 1 min of source video ---")
    print(f"[measure_subtitle_cost] face-detect/reframe : {args.reframe_sec_per_min:.3f} sec/min -> ${reframe_cost_usd_per_min:.8f} ({reframe_cost_usd_per_min * args.jpy_rate:.5f}円)")
    print(f"[measure_subtitle_cost] subtitle burn-in     : {subtitle_sec_per_min:.3f} sec/min -> ${subtitle_cost_usd_per_min:.8f} ({subtitle_cost_usd_per_min * args.jpy_rate:.5f}円)")
    print(f"[measure_subtitle_cost] TOTAL                 : {total_sec_per_min:.3f} sec/min -> ${total_cost_usd_per_min:.8f} ({total_cost_usd_per_min * args.jpy_rate:.5f}円)")


if __name__ == "__main__":
    main()
