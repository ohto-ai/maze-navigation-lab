#!/usr/bin/env python3
"""Run B against A for many seeds and summarize all feedback modes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
A_PATH = ROOT / "a_maze.py"
B_PATH = ROOT / "b_agent.py"

MODES = ("none", "distance", "direction", "shortest")
STRATEGY_FOR_MODE = {
    "none": "s0",
    "distance": "s1",
    "direction": "s2",
    "shortest": "s3",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-compare B's strategies over many maze seeds"
    )
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=int,
        default=100,
        help="number of consecutive seeds (seed_start .. seed_start+seeds-1)",
    )
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="comma-separated subset of none,distance,direction,shortest",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="per-response timeout passed to B",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="communication retries in B"
    )
    parser.add_argument(
        "--process-timeout",
        type=float,
        default=120.0,
        help="wall-clock timeout for one B+A run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "benchmark_10x10.json",
        help="where to write machine-readable aggregate results (JSON)",
    )
    parser.add_argument(
        "--visualize-seed",
        type=int,
        default=None,
        help="run that seed once with A and B visualization instead of the batch",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def selected_modes(text: str) -> list[str]:
    modes = [part.strip() for part in text.split(",") if part.strip()]
    bad = [mode for mode in modes if mode not in MODES]
    if bad:
        raise ValueError(f"unknown modes: {', '.join(bad)}")
    if not modes:
        raise ValueError("no modes selected")
    return modes


def a_command(size: int, mode: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(A_PATH),
        "--size",
        str(size),
        "--mode",
        mode,
        "--seed",
        str(seed),
    ]


def b_command(
    mode: str, args: argparse.Namespace, seed: int, visualize: bool = False
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(B_PATH),
        "--mode",
        mode,
        "--strategy",
        "auto",
        "--max-steps",
        str(args.max_steps),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--a-command",
        json.dumps(a_command(args.size, mode, seed), ensure_ascii=True),
    ]
    if visualize:
        command += ["--visualize"]
    else:
        command.append("--quiet")
    return command


def parse_b_report(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "mode" in value and "steps" in value:
            return value
    raise ValueError(f"B did not emit a valid report; stdout was: {stdout!r}")


def run_one(mode: str, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            b_command(mode, args, seed),
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.process_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "mode": mode,
            "strategy": STRATEGY_FOR_MODE[mode],
            "steps": 0,
            "reached": False,
            "explored_cells": 0,
            "time": round(time.perf_counter() - started, 6),
            "error": f"process timeout: {exc}",
        }

    try:
        report = parse_b_report(completed.stdout)
    except ValueError as exc:
        report = {
            "mode": mode,
            "strategy": STRATEGY_FOR_MODE[mode],
            "steps": 0,
            "reached": False,
            "explored_cells": 0,
            "time": round(time.perf_counter() - started, 6),
            "error": str(exc),
            "stderr": completed.stderr[-2000:],
        }
    report["seed"] = seed
    report["process_returncode"] = completed.returncode
    return report


def summarize(
    runs: list[dict[str, Any]],
    modes: list[str],
    size: int,
    seeds: list[int],
    max_steps: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        subset = [run for run in runs if run["mode"] == mode]
        reached = [run for run in subset if run.get("reached")]
        steps = [run["steps"] for run in reached]
        explored = [run["explored_cells"] for run in reached]
        times = [run["time"] for run in subset]
        rows.append(
            {
                "mode": mode,
                "strategy": STRATEGY_FOR_MODE[mode],
                "runs": len(subset),
                "reached": len(reached),
                "reach_rate": round(len(reached) / len(subset), 4) if subset else 0.0,
                "avg_steps": round(statistics.mean(steps), 2) if steps else None,
                "median_steps": statistics.median(steps) if steps else None,
                "min_steps": min(steps) if steps else None,
                "max_steps": max(steps) if steps else None,
                "avg_explored_cells": (
                    round(statistics.mean(explored), 2) if explored else None
                ),
                "avg_time": round(statistics.mean(times), 4) if times else None,
            }
        )
    return {
        "config": {
            "size": size,
            "seed_start": seeds[0] if seeds else None,
            "seed_count": len(seeds),
            "seeds": seeds,
            "modes": modes,
            "max_steps": max_steps,
        },
        "summary": rows,
        "runs": runs,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        "| mode | strategy | runs | reached | avg steps | avg explored | "
        "median steps | avg time (s) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["avg_steps"] is None:
            steps = "-"
            explored = "-"
            median = "-"
        else:
            steps = f"{row['avg_steps']:.2f}"
            explored = f"{row['avg_explored_cells']:.2f}"
            median = f"{row['median_steps']:.1f}"
        mode_label = row["mode"]
        if mode_label == "shortest":
            mode_label += " (oracle)"
        print(
            f"| {mode_label} | {row['strategy']} | {row['runs']} | "
            f"{row['reached']}/{row['runs']} | {steps} | {explored} | "
            f"{median} | {row['avg_time']:.4f} |"
        )
    print()


def run_visualization(args: argparse.Namespace, mode: str) -> int:
    command = b_command(mode, args, args.visualize_seed, visualize=True)
    # A visualization is enabled via B's custom A command.  Rebuild it with
    # A's --visualize flag while preserving the same A configuration.
    a_cmd = a_command(args.size, mode, args.visualize_seed) + ["--visualize"]
    idx = command.index("--a-command")
    command[idx + 1] = json.dumps(a_cmd, ensure_ascii=True)
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.process_timeout,
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        modes = selected_modes(args.modes)
    except ValueError as exc:
        print(f"run_experiment: {exc}", file=sys.stderr)
        return 2
    if args.seeds <= 0:
        print("run_experiment: --seeds must be positive", file=sys.stderr)
        return 2

    if args.visualize_seed is not None:
        return run_visualization(args, modes[0])

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for mode in modes:
            report = run_one(mode, args, seed)
            runs.append(report)
            if args.verbose or not report.get("reached"):
                suffix = ""
                if report.get("error"):
                    suffix = f" error={report['error']}"
                elif report.get("reason"):
                    suffix = f" reason={report['reason']}"
                print(
                    f"seed={seed:>4} mode={mode:<9} strategy={report.get('strategy')} "
                    f"steps={report['steps']} explored={report['explored_cells']} "
                    f"reached={report['reached']}{suffix}",
                    file=sys.stderr,
                )

    result = summarize(runs, modes, args.size, seeds, args.max_steps)
    print_table(result["summary"])

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"results written to {args.output}")
    except OSError as exc:
        print(f"run_experiment: cannot write {args.output}: {exc}", file=sys.stderr)
        return 2

    failed = [
        run
        for run in runs
        if not run.get("reached")
        or run.get("error")
        or run.get("process_returncode", 0) != 0
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
