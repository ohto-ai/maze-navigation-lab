"""Core regression tests for the maze navigation lab.

Runs without a display: the Tkinter visualization itself is not started.  The
GUI trace builder is exercised only when tkinter is importable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a_maze import DIRS, PATH, MazeEnvironment, shortest_distances  # noqa: E402
from b_agent import (  # noqa: E402
    DFSStrategy,
    FrontierStrategy,
    GradientStrategy,
    Navigator,
)

MODES = ("none", "distance", "direction", "shortest")
STRATEGY = {
    "none": "s0",
    "distance": "s1",
    "direction": "s2",
    "shortest": "s3",
}


class LocalClient:
    def __init__(self, env: MazeEnvironment) -> None:
        self.env = env

    def move(self, direction: str) -> dict[str, object]:
        return self.env.move(direction)


def path_cells(env: MazeEnvironment) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(env.size)
        for c in range(env.size)
        if env.maze[r][c] == PATH
    ]


class MazeTests(unittest.TestCase):
    def test_perfect_maze_invariants(self) -> None:
        for size in (4, 7, 10, 14):
            for seed in (100, 101, 102):
                with self.subTest(size=size, seed=seed):
                    env = MazeEnvironment(size=size, mode="distance", seed=seed)
                    cells = set(path_cells(env))
                    self.assertGreaterEqual(len(cells), 2)
                    self.assertIn(env.start, cells)
                    self.assertIn(env.goal, cells)
                    self.assertNotEqual(env.start, env.goal)

                    seen = {env.start}
                    queue = deque([env.start])
                    while queue:
                        r, c = queue.popleft()
                        for dr, dc in DIRS.values():
                            nb = (r + dr, c + dc)
                            if nb in cells and nb not in seen:
                                seen.add(nb)
                                queue.append(nb)
                    self.assertEqual(seen, cells)

                    edges = 0
                    for r, c in cells:
                        if (r + 1, c) in cells:
                            edges += 1
                        if (r, c + 1) in cells:
                            edges += 1
                    self.assertEqual(edges, len(cells) - 1)
                    self.assertEqual(set(env.shortest_map), cells)
                    self.assertEqual(env.shortest_map[env.goal], 0)

    def test_seed_is_reproducible_and_mode_independent(self) -> None:
        layouts = []
        for mode in MODES:
            env = MazeEnvironment(size=14, mode=mode, seed=101)
            layouts.append((env.maze, env.start, env.goal))
        self.assertTrue(all(layout == layouts[0] for layout in layouts))

        again = MazeEnvironment(size=14, mode="distance", seed=101)
        self.assertEqual(
            (again.maze, again.start, again.goal), layouts[0]
        )

    def test_feedback_contracts(self) -> None:
        forbidden = {
            "car",
            "car_pos",
            "goal",
            "goal_pos",
            "start_pos",
            "maze",
            "path",
        }
        for mode in MODES:
            with self.subTest(mode=mode):
                env = MazeEnvironment(size=10, mode=mode, seed=101)
                response = env.move("up")
                self.assertIs(type(response["success"]), bool)
                self.assertIs(type(response["is_goal"]), bool)
                self.assertFalse(forbidden.intersection(response))

                if mode == "none":
                    self.assertEqual(set(response), {"success", "is_goal"})
                elif mode == "distance":
                    self.assertIn("distance", response)
                    self.assertNotIn("shortest_distance", response)
                    self.assertEqual(
                        response["distance"],
                        abs(env.car[0] - env.goal[0])
                        + abs(env.car[1] - env.goal[1]),
                    )
                elif mode == "direction":
                    self.assertIn("dir_hint", response)
                    self.assertNotIn("distance", response)
                    hint = response["dir_hint"]
                    self.assertTrue(
                        hint is None
                        or hint in {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
                    )
                else:
                    self.assertIn("shortest_distance", response)
                    self.assertNotIn("distance", response)
                    self.assertEqual(
                        response["shortest_distance"],
                        env.shortest_map[env.car],
                    )

    def test_shortest_distance_matches_bfs(self) -> None:
        env = MazeEnvironment(size=14, mode="shortest", seed=101)
        expected = shortest_distances(env.maze, env.goal)
        self.assertEqual(env.shortest_map, expected)

    def test_gradient_strategy_reaches(self) -> None:
        for size in (4, 10, 14):
            for seed in (100, 101, 102):
                with self.subTest(size=size, seed=seed):
                    env = MazeEnvironment(
                        size=size, mode="shortest", seed=seed
                    )
                    agent = Navigator(
                        LocalClient(env), mode="shortest", max_steps=100_000
                    )
                    GradientStrategy(agent).run()
                    self.assertTrue(agent.reached)
                    self.assertEqual(agent.steps, env.step_count)
                    self.assertGreaterEqual(agent.explored_cells, 1)

    def test_fair_strategies_reach_small_batch(self) -> None:
        strategies = {
            "none": DFSStrategy,
            "distance": FrontierStrategy,
            "direction": FrontierStrategy,
        }
        for mode, strategy in strategies.items():
            for seed in (100, 101):
                with self.subTest(mode=mode, seed=seed):
                    env = MazeEnvironment(size=10, mode=mode, seed=seed)
                    agent = Navigator(
                        LocalClient(env), mode=mode, max_steps=100_000
                    )
                    strategy(agent).run()
                    self.assertTrue(agent.reached)
                    self.assertEqual(agent.steps, env.step_count)


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def a_command(mode: str, seed: int) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(ROOT / "a_maze.py"),
            "--size",
            "10",
            "--mode",
            mode,
            "--seed",
            str(seed),
        ]

    def test_duplicate_request_id_is_idempotent(self) -> None:
        input_lines = "\n".join(
            [
                '{"cmd":"move","dir":"up","id":1}',
                '{"cmd":"move","dir":"up","id":1}',
                '{"cmd":"state","id":2}',
                '{"cmd":"quit"}',
            ]
        )
        completed = subprocess.run(
            self.a_command("none", 101),
            input=input_lines + "\n",
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = [
            json.loads(line)
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], lines[1])
        self.assertEqual(lines[2]["step_count"], 1)

    def test_b_end_to_end_all_modes(self) -> None:
        for mode in MODES:
            with self.subTest(mode=mode):
                a_cmd = self.a_command(mode, 101)
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / "b_agent.py"),
                        "--mode",
                        mode,
                        "--strategy",
                        "auto",
                        "--max-steps",
                        "10000",
                        "--timeout",
                        "5",
                        "--retries",
                        "1",
                        "--quiet",
                        "--a-command",
                        json.dumps(a_cmd),
                    ],
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stderr
                )
                reports = [
                    json.loads(line)
                    for line in completed.stdout.splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(reports), 1)
                report = reports[0]
                self.assertEqual(report["mode"], mode)
                self.assertEqual(report["strategy"], STRATEGY[mode])
                self.assertTrue(report["reached"], report)
                self.assertGreater(report["steps"], 0)
                self.assertGreaterEqual(report["explored_cells"], 1)
                for key in ("car", "car_pos", "goal", "goal_pos"):
                    self.assertNotIn(key, report)


class VisualizationDataTests(unittest.TestCase):
    def test_gui_trace_builder_covers_all_modes(self) -> None:
        try:
            import gui_demo
        except ImportError as exc:  # tkinter unavailable on some CI images
            self.skipTest(f"tkinter unavailable: {exc}")

        traces = gui_demo.build_traces(10, 100)
        self.assertEqual(set(traces), set(MODES))
        layouts = {
            (trace.maze, trace.start, trace.goal)
            for trace in traces.values()
        }
        self.assertEqual(len(layouts), 1)
        for mode, trace in traces.items():
            self.assertTrue(trace.reached, mode)
            self.assertEqual(
                len(trace.gradient), len(path_cells(
                    MazeEnvironment(size=10, mode="distance", seed=100)
                ))
            )


if __name__ == "__main__":
    unittest.main()
