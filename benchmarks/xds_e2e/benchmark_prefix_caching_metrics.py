# SPDX-License-Identifier: Apache-2.0
"""Run vLLM's prefix benchmark and emit per-request latency metrics."""

# Standard
import json
from pathlib import Path
import sys
import time


VLLM_BENCHMARKS = Path("/home/yangxc/workspace/vllm-v0180/benchmarks")
sys.path.insert(0, str(VLLM_BENCHMARKS))

# Third Party
import benchmark_prefix_caching as benchmark  # noqa: E402


def repeat_in_cycles(requests, repeat_count, sort=False):
    """Keep one cold cycle followed by repeat cycles in a stable order."""
    del sort
    return [request.prompt for _ in range(repeat_count) for request in requests]


def main() -> None:
    parser = benchmark.create_argument_parser()
    args = parser.parse_args()

    def test_prefix(llm=None, sampling_params=None, prompts=None):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        wall_seconds = time.perf_counter() - started

        requests = []
        for index, output in enumerate(outputs):
            metrics = output.metrics
            if metrics is None:
                raise RuntimeError("vLLM request metrics are unavailable")
            requests.append(
                {
                    "index": index,
                    "phase": "cold" if index < args.num_prompts else "hit",
                    "inference_ms": (
                        metrics.last_token_ts - metrics.scheduled_ts
                    )
                    * 1000.0,
                    "queue_ms": (metrics.scheduled_ts - metrics.queued_ts) * 1000.0,
                    "ttft_ms": metrics.first_token_latency * 1000.0,
                    "num_cached_tokens": output.num_cached_tokens,
                }
            )

        print(f"cost time {wall_seconds}")
        print(
            "[PREFIX_BENCH_METRICS] "
            + json.dumps(
                {
                    "num_unique_prompts": args.num_prompts,
                    "repeat_count": args.repeat_count,
                    "wall_seconds": wall_seconds,
                    "requests": requests,
                },
                separators=(",", ":"),
            )
        )

    benchmark.repeat_and_sort_requests = repeat_in_cycles
    benchmark.test_prefix = test_prefix
    benchmark.main(args)


if __name__ == "__main__":
    main()
