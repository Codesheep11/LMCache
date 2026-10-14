# SPDX-License-Identifier: Apache-2.0
"""Run vLLM's prefix benchmark and emit per-request latency metrics."""

# Standard
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


VLLM_BENCHMARKS = Path("/home/yangxc/workspace/vllm-v0180/benchmarks")
sys.path.insert(0, str(VLLM_BENCHMARKS))

# Third Party
import benchmark_prefix_caching as benchmark  # noqa: E402


def load_prompt_trace(path: Path, count: int, input_length_range: str):
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported prompt trace schema in {path}")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != count:
        raise ValueError(f"{path} must contain exactly {count} prompts")
    min_len, max_len = map(int, input_length_range.split(":"))
    requests = []
    for index, token_ids in enumerate(prompts):
        if not isinstance(token_ids, list) or not min_len <= len(token_ids) <= max_len:
            raise ValueError(
                f"prompt {index} in {path} is outside {min_len}:{max_len}"
            )
        if not all(isinstance(token, int) and token >= 0 for token in token_ids):
            raise ValueError(f"prompt {index} in {path} contains an invalid token id")
        requests.append(
            benchmark.Request(
                benchmark.TokensPrompt(prompt_token_ids=token_ids),
                len(token_ids),
                1,
            )
        )
    return requests


def repeat_in_cycles(requests, repeat_count, sort=False):
    """Keep one cold cycle followed by repeat cycles in a stable order."""
    del sort
    return [request.prompt for _ in range(repeat_count) for request in requests]


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = benchmark.create_argument_parser()
        parser.add_argument("--prompt-trace")
        parser.add_argument("--result-file")
        parser.add_argument("--simultaneous-replay", action="store_true")
        parser.add_argument("--shuffle-all-replays", action="store_true")
        parser.add_argument("--replay-shuffle-seed", type=int, default=0)
        args = parser.parse_args()

    replay_order = []

    def prepare_replay(requests, repeat_count, sort=False):
        """Expand requests into reproducible, optionally shuffled cycles."""
        nonlocal replay_order
        del sort
        expanded = []
        for cycle in range(repeat_count):
            expanded.extend(
                {
                    "copy_index": cycle,
                    "prompt_index": prompt_index,
                    "prompt": request.prompt,
                }
                for prompt_index, request in enumerate(requests)
            )
        if args.shuffle_all_replays:
            random.Random(args.replay_shuffle_seed).shuffle(expanded)

        occurrence_counts = {index: 0 for index in range(len(requests))}
        replay_prompts = []
        for item in expanded:
            prompt_index = item["prompt_index"]
            occurrence = occurrence_counts[prompt_index]
            occurrence_counts[prompt_index] += 1
            replay_order.append(
                {
                    "cycle": occurrence,
                    "copy_index": item["copy_index"],
                    "prompt_index": prompt_index,
                }
            )
            replay_prompts.append(item["prompt"])
        return replay_prompts

    def test_prefix(llm=None, sampling_params=None, prompts=None):
        unique_prompts = prompts[: args.num_prompts]
        hit_index_sequence = None
        raw_hit_index_sequence = os.environ.get(
            "LMCACHE_BENCH_HIT_INDEX_SEQUENCE", ""
        ).strip()
        if raw_hit_index_sequence:
            hit_index_sequence = [
                int(value.strip())
                for value in raw_hit_index_sequence.split(",")
                if value.strip()
            ]
            if not hit_index_sequence or any(
                index < 0 or index >= len(unique_prompts)
                for index in hit_index_sequence
            ):
                raise ValueError(
                    "LMCACHE_BENCH_HIT_INDEX_SEQUENCE must contain valid "
                    "zero-based unique-prompt indices"
                )
        hit_prompts = prompts[args.num_prompts :]
        if hit_index_sequence is not None:
            hit_prompts = [unique_prompts[index] for index in hit_index_sequence]

        warmup_input_len = int(os.environ.get("LMCACHE_BENCH_WARMUP_INPUT_LEN", "0"))
        warmup_settle_seconds = float(
            os.environ.get("LMCACHE_BENCH_WARMUP_SETTLE_SECONDS", "0")
        )
        if warmup_input_len > 0:
            warmup_ids = [
                22000 + ((index * 7919 + 104729) % 10000)
                for index in range(warmup_input_len)
            ]
            llm.generate(
                [benchmark.TokensPrompt(prompt_token_ids=warmup_ids)],
                sampling_params=sampling_params,
            )
            if warmup_settle_seconds > 0:
                time.sleep(warmup_settle_seconds)

        total_started = time.perf_counter()
        if args.simultaneous_replay:
            expected_request_count = args.num_prompts * args.repeat_count
            if len(prompts) != expected_request_count:
                raise RuntimeError(
                    f"expected {expected_request_count} replay prompts, "
                    f"received {len(prompts)}"
                )
            batch_start_ns = time.monotonic_ns()
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            batch_end_ns = time.monotonic_ns()
            if len(outputs) != expected_request_count:
                raise RuntimeError(
                    f"expected {expected_request_count} outputs, "
                    f"received {len(outputs)}"
                )
            total_wall_seconds = time.perf_counter() - total_started
            cold_wall_seconds = None
            hit_wall_seconds = None
            hit_started_ns = None
            hit_ended_ns = None
            cycle_wall_seconds = None
        else:
            cold_started = time.perf_counter()
            cold_outputs = llm.generate(
                unique_prompts, sampling_params=sampling_params
            )
            cold_wall_seconds = time.perf_counter() - cold_started

            cold_settle_seconds = float(
                os.environ.get("LMCACHE_BENCH_COLD_SETTLE_SECONDS", "0")
            )
            if cold_settle_seconds > 0:
                time.sleep(cold_settle_seconds)
            if os.environ.get("LMCACHE_BENCH_DROP_CACHE_AFTER_COLD") == "1":
                time.sleep(3.0)
                os.sync()
                Path("/proc/sys/vm/drop_caches").write_text("3\n")

            hit_started_ns = time.monotonic_ns()
            hit_started = time.perf_counter()
            hit_outputs = llm.generate(hit_prompts, sampling_params=sampling_params)
            hit_wall_seconds = time.perf_counter() - hit_started
            hit_ended_ns = time.monotonic_ns()
            total_wall_seconds = time.perf_counter() - total_started
            outputs = cold_outputs + hit_outputs
            batch_start_ns = None
            batch_end_ns = None
            cycle_wall_seconds = None

        requests = []
        for flat_index, output in enumerate(outputs):
            metrics = output.metrics
            if metrics is None:
                raise RuntimeError("vLLM request metrics are unavailable")
            replay_item = (
                replay_order[flat_index]
                if len(replay_order) == len(outputs)
                else {
                    "cycle": flat_index // args.num_prompts,
                    "prompt_index": flat_index % args.num_prompts,
                }
            )
            cycle = replay_item["cycle"]
            requests.append(
                {
                    "cycle": cycle,
                    "index": replay_item["prompt_index"],
                    "copy_index": replay_item.get("copy_index", cycle),
                    "submission_index": flat_index,
                    "phase": "cold" if cycle == 0 else "hit",
                    "inference_ms": (
                        metrics.last_token_ts - metrics.scheduled_ts
                    )
                    * 1000.0,
                    "scheduled_ns": round(metrics.scheduled_ts * 1_000_000_000),
                    "first_token_ns": round(
                        metrics.first_token_ts * 1_000_000_000
                    ),
                    "last_token_ns": round(
                        metrics.last_token_ts * 1_000_000_000
                    ),
                    "queue_ms": (metrics.scheduled_ts - metrics.queued_ts) * 1000.0,
                    "ttft_ms": metrics.first_token_latency * 1000.0,
                    "prefill_latency_ms": metrics.first_token_latency * 1000.0,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "num_cached_tokens": output.num_cached_tokens,
                }
            )

        # With max_tokens=1, request arrival-to-first-token is the end-to-end
        # prefill latency (queue + lookup/restore + residual compute + the one
        # sampled token).  The experiment-wide cost is the sum across every
        # request in every cycle, including the first/cold cycle.
        cost_time_seconds = sum(
            request["prefill_latency_ms"] for request in requests
        ) / 1000.0
        print(f"cost time {cost_time_seconds}")
        result = {
            "tp_size": args.tensor_parallel_size,
            "max_num_seqs": args.max_num_seqs,
            "enable_prefix_caching": args.enable_prefix_caching,
            "num_unique_prompts": args.num_prompts,
            "repeat_count": args.repeat_count,
            "prompt_trace": (
                str(Path(args.prompt_trace).resolve())
                if args.prompt_trace is not None
                else None
            ),
            "prompt_trace_sha256": (
                hashlib.sha256(Path(args.prompt_trace).read_bytes()).hexdigest()
                if args.prompt_trace is not None
                else None
            ),
            "hit_prompt_indices": hit_index_sequence,
            "warmup_input_len_excluded": warmup_input_len,
            "warmup_settle_seconds_excluded": warmup_settle_seconds,
            "cost_time_seconds": cost_time_seconds,
            "cost_time_definition": (
                "sum(request arrival-to-first-token latency) over all "
                "output_len=1 requests, including queue and cold cycle"
            ),
            "wall_seconds": total_wall_seconds,
            "cold_wall_seconds": cold_wall_seconds,
            "hit_wall_seconds": hit_wall_seconds,
            "hit_wall_start_ns": hit_started_ns,
            "hit_wall_end_ns": hit_ended_ns,
            "batch_wall_start_ns": batch_start_ns,
            "batch_wall_end_ns": batch_end_ns,
            "cycle_wall_seconds": cycle_wall_seconds,
            "submission_mode": (
                f"simultaneous_{args.num_prompts * args.repeat_count}_request_batch"
                if args.simultaneous_replay
                else "split_cold_and_replay_batches"
            ),
            "replay_order_mode": (
                "globally_shuffled_all_requests"
                if args.shuffle_all_replays
                else "stable_within_cycles"
            ),
            "replay_shuffle_seed": (
                args.replay_shuffle_seed if args.shuffle_all_replays else None
            ),
            "replay_order": replay_order,
            "replay_order_sha256": hashlib.sha256(
                json.dumps(replay_order, separators=(",", ":")).encode()
            ).hexdigest(),
            "total_wall_seconds": total_wall_seconds,
            "requests": requests,
        }
        print("[PREFIX_BENCH_METRICS] " + json.dumps(result, separators=(",", ":")))
        if args.result_file is not None:
            result_file = Path(args.result_file)
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(result, indent=2) + "\n")

    if args.output_len != 1:
        raise ValueError("offline prefill experiments require --output-len 1")
    if args.prompt_trace is not None:
        traced_requests = load_prompt_trace(
            Path(args.prompt_trace), args.num_prompts, args.input_length_range
        )

        def sample_trace_requests(**kwargs):
            del kwargs
            return traced_requests

        benchmark.sample_requests_from_random = sample_trace_requests
    benchmark.repeat_and_sort_requests = prepare_replay
    benchmark.test_prefix = test_prefix
    benchmark.main(args)


if __name__ == "__main__":
    parser = benchmark.create_argument_parser()
    parser.add_argument("--prompt-trace")
    parser.add_argument("--result-file")
    parser.add_argument("--simultaneous-replay", action="store_true")
    parser.add_argument("--shuffle-all-replays", action="store_true")
    parser.add_argument("--replay-shuffle-seed", type=int, default=0)
    main_args = parser.parse_args()
    main(main_args)
