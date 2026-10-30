import argparse
import json
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def summarize(samples):
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "p95_nearest_rank_ms": ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)],
        "mean_ms": statistics.mean(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure full local HTTP retrieval latency including query embedding."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--question",
        default="How does asyncio.create_task schedule a coroutine?",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/http_retrieval_latency_v1.json")
    args = parser.parse_args()
    if args.top_k <= 0 or args.warmups < 0 or args.repetitions < 10:
        parser.error("top-k must be positive, warmups nonnegative, repetitions >= 10.")

    modes = ("dense", "hybrid")
    try:
        with httpx.Client(
            base_url=args.base_url.rstrip("/"),
            timeout=httpx.Timeout(connect=5, read=60, write=5, pool=5),
        ) as client:
            readiness = client.get("/ready")
            readiness.raise_for_status()
            ready_payload = readiness.json()
            if not ready_payload["ready"]:
                parser.error(f"Backend index is not ready: {ready_payload['detail']}")

            def measure(mode):
                started = perf_counter()
                response = client.post(
                    "/retrieve",
                    json={
                        "question": args.question,
                        "mode": mode,
                        "top_k": args.top_k,
                    },
                )
                elapsed_ms = (perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()
                if payload["index_version"] != ready_payload["index_version"]:
                    raise RuntimeError("Active index changed during HTTP measurements.")
                return elapsed_ms

            for _ in range(args.warmups):
                for mode in modes:
                    measure(mode)
            raw = {mode: [] for mode in modes}
            for repetition in range(args.repetitions):
                order = list(modes)
                random.Random(20261020 + repetition).shuffle(order)
                for mode in order:
                    raw[mode].append(measure(mode))
    except (httpx.HTTPError, RuntimeError, KeyError) as error:
        parser.error(f"Could not measure HTTP retrieval: {error}")

    report = {
        "report_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "index_version": ready_payload["index_version"],
        "question": args.question,
        "top_k": args.top_k,
        "warmup_requests_per_mode": args.warmups,
        "timed_requests_per_mode": args.repetitions,
        "randomized_order_seed": 20261020,
        "conditions": {
            "same_loaded_api_process_and_index": True,
            "includes": [
                "HTTP loopback request/response",
                "request validation and response serialization",
                "query embedding",
                "dense or hybrid retrieval",
            ],
            "excludes": [
                "process/model startup",
                "index preparation/rebuild",
                "LLM generation",
                "browser rendering",
            ],
            "query_embedding_latency_separate_measurement": False,
        },
        "hardware": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "latency": {
            mode: {**summarize(samples), "raw_ms": samples}
            for mode, samples in raw.items()
        },
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for mode in modes:
        values = report["latency"][mode]
        print(
            f"{mode}: HTTP retrieval median/p95="
            f"{values['median_ms']:.2f}/{values['p95_nearest_rank_ms']:.2f} ms"
        )
    print(f"Raw HTTP measurements: {report_path}")


if __name__ == "__main__":
    main()
