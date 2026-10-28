#!/usr/bin/env python3
"""A-side maze environment.

Generates a perfect maze, places a car and a goal, and speaks a JSON-lines
protocol on stdin/stdout.  Every stdout line is exactly one protocol response;
human-readable logging and optional visualization go to stderr.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict, deque
from typing import Any

WALL = 1
PATH = 0

DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def shortest_distances(
    maze: list[list[int]], goal: tuple[int, int]
) -> dict[tuple[int, int], int]:
    """BFS distance (in moves) from every passage cell to the goal.

    This is the leaky "gradient" signal used by the optional shortest mode;
    it is deliberately *not* part of the three fair feedback modes.
    """
    n = len(maze)
    distances = {goal: 0}
    queue: deque[tuple[int, int]] = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if maze[nr][nc] == WALL:
                continue
            cell = (nr, nc)
            if cell not in distances:
                distances[cell] = distances[(r, c)] + 1
                queue.append(cell)
    return distances


def direction_hint(
    car: tuple[int, int], goal: tuple[int, int], style: str = "8"
) -> str | None:
    """Absolute 8-way (or 4-way) octant from the car to the goal.

    The hint deliberately reveals only a sign quadrant, never magnitudes or
    absolute coordinates.  At the goal there is no meaningful direction.
    """
    dr = _sign(goal[0] - car[0])
    dc = _sign(goal[1] - car[1])
    if dr == 0 and dc == 0:
        return None

    if style == "8":
        if dr < 0:
            return "N" if dc == 0 else ("NE" if dc > 0 else "NW")
        if dr > 0:
            return "S" if dc == 0 else ("SE" if dc > 0 else "SW")
        return "E" if dc > 0 else "W"

    # Four-way hint: choose the dominant axis; ties prefer the vertical axis.
    if abs(goal[0] - car[0]) >= abs(goal[1] - car[1]):
        return "S" if dr > 0 else "N"
    return "E" if dc > 0 else "W"


class MazeEnvironment:
    """Authoritative maze state.  Only normal feedback is sent over stdout."""

    def __init__(
        self,
        size: int = 10,
        mode: str = "distance",
        seed: int = 42,
        hint_style: str = "8",
    ) -> None:
        if size < 4:
            raise ValueError("size must be at least 4")
        if mode not in {"none", "distance", "direction", "shortest"}:
            raise ValueError(f"unknown feedback mode: {mode}")
        if hint_style not in {"4", "8"}:
            raise ValueError(f"unknown hint style: {hint_style}")

        self.size = size
        self.mode = mode
        self.seed = seed
        self.hint_style = hint_style
        self.rng = random.Random(seed)
        self.maze = self._generate_maze()

        path_cells = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if self.maze[r][c] == PATH
        ]
        if len(path_cells) < 2:
            raise ValueError("maze does not contain two distinct path cells")
        self.start = self.rng.choice(path_cells)
        remaining = [cell for cell in path_cells if cell != self.start]
        self.goal = self.rng.choice(remaining)
        self.shortest_map = shortest_distances(self.maze, self.goal)

        self.car = self.start
        self.step_count = 0
        self.path: list[tuple[int, int]] = [self.start]

    # ------------------------------------------------------------------ maze
    def _generate_maze(self) -> list[list[int]]:
        """Recursive backtracker on odd-coordinate rooms.

        Walls between rooms are carved only while growing a spanning tree, so
        the resulting passage graph is connected and acyclic (a perfect maze).
        """
        n = self.size
        maze = [[WALL] * n for _ in range(n)]
        rooms = [(r, c) for r in range(1, n, 2) for c in range(1, n, 2)]
        if not rooms:
            raise ValueError("size is too small to contain a room")

        first = self.rng.choice(rooms)
        maze[first[0]][first[1]] = PATH
        visited = {first}
        stack = [first]

        while stack:
            r, c = stack[-1]
            candidates = []
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                    # Only odd-coordinate room cells are legal.
                    if nr % 2 == 1 and nc % 2 == 1:
                        candidates.append((nr, nc, r + dr // 2, c + dc // 2))

            if not candidates:
                stack.pop()
                continue

            nr, nc, wr, wc = self.rng.choice(candidates)
            maze[wr][wc] = PATH
            maze[nr][nc] = PATH
            visited.add((nr, nc))
            stack.append((nr, nc))

        return maze

    # -------------------------------------------------------------- feedback
    def _feedback(self, success: bool) -> dict[str, Any]:
        response: dict[str, Any] = {
            "success": success,
            "is_goal": self.car == self.goal,
        }
        if self.mode == "distance":
            response["distance"] = abs(self.car[0] - self.goal[0]) + abs(
                self.car[1] - self.goal[1]
            )
        elif self.mode == "direction":
            response["dir_hint"] = direction_hint(
                self.car, self.goal, self.hint_style
            )
        elif self.mode == "shortest":
            response["shortest_distance"] = self.shortest_map[self.car]
        return response

    # --------------------------------------------------------------- actions
    def move(self, direction: str) -> dict[str, Any]:
        if direction not in DIRS:
            raise ValueError(f"invalid direction: {direction!r}")

        self.step_count += 1
        dr, dc = DIRS[direction]
        nr, nc = self.car[0] + dr, self.car[1] + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return self._feedback(False)
        if self.maze[nr][nc] == WALL:
            return self._feedback(False)

        self.car = (nr, nc)
        self.path.append(self.car)
        return self._feedback(True)

    def reset(self) -> dict[str, Any]:
        self.car = self.start
        self.step_count = 0
        self.path = [self.start]
        response = self._feedback(True)
        response["reset"] = True
        return response

    def state(self) -> dict[str, Any]:
        """Debug/evaluation snapshot.  The B program must never call this."""
        return {
            "debug": True,
            "mode": self.mode,
            "maze_size": self.size,
            "maze": [row[:] for row in self.maze],
            "car_pos": list(self.car),
            "start_pos": list(self.start),
            "goal_pos": list(self.goal),
            "step_count": self.step_count,
            "path": [list(cell) for cell in self.path],
            "is_goal": self.car == self.goal,
        }

    # ------------------------------------------------------------ rendering
    def render(self) -> str:
        path_cells = set(self.path)
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                cell = (r, c)
                if cell == self.car:
                    row.append("C")
                elif cell == self.goal:
                    row.append("G")
                elif cell == self.start:
                    row.append("S")
                elif cell in path_cells:
                    row.append("*")
                elif self.maze[r][c] == WALL:
                    row.append("#")
                else:
                    row.append(".")
            lines.append("".join(row))
        return "\n".join(lines)


def encode(response: dict[str, Any]) -> str:
    return json.dumps(response, ensure_ascii=True, separators=(",", ":"))


def serve(env: MazeEnvironment, visualize: bool) -> int:
    if visualize:
        print(
            f"A: size={env.size} mode={env.mode} seed={env.seed} "
            f"start={env.start} goal={env.goal}",
            file=sys.stderr,
        )
        print(env.render(), file=sys.stderr, flush=True)

    cache: OrderedDict[int | str, dict[str, Any]] = OrderedDict()
    should_quit = False

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            print(encode({"error": f"invalid request: {exc}"}), flush=True)
            continue

        req_id = request.get("id")
        if req_id is not None and not isinstance(req_id, (int, str)):
            print(encode({"error": "id must be an integer or string"}), flush=True)
            continue

        if req_id is not None and req_id in cache:
            print(encode(cache[req_id]), flush=True)
            continue

        cmd = request.get("cmd")
        try:
            if cmd == "move":
                direction = request.get("dir")
                if not isinstance(direction, str) or direction not in DIRS:
                    response = {"error": "move requires dir in up/down/left/right"}
                else:
                    response = env.move(direction)
            elif cmd == "reset":
                response = env.reset()
            elif cmd == "state":
                response = env.state()
            elif cmd == "quit":
                response = {"ok": True, "bye": True}
                should_quit = True
            else:
                response = {"error": f"unknown cmd: {cmd!r}"}
        except ValueError as exc:
            response = {"error": str(exc)}

        if req_id is not None:
            response["id"] = req_id
            cache[req_id] = response
            if len(cache) > 1024:
                cache.popitem(last=False)

        print(encode(response), flush=True)
        if should_quit:
            break

    if visualize:
        print(
            f"final: car={env.car} goal={env.goal} steps={env.step_count}",
            file=sys.stderr,
        )
        print(env.render(), file=sys.stderr, flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A-side perfect-maze environment (JSON lines over stdio)"
    )
    parser.add_argument("--size", type=int, default=10, help="maze dimension")
    parser.add_argument(
        "--mode",
        choices=("none", "distance", "direction", "shortest"),
        default="distance",
        help="feedback mode (shortest is the leaky gradient/oracle control)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--hint-style",
        choices=("4", "8"),
        default="8",
        help="direction hint granularity",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="render the maze to stderr at startup and shutdown",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        env = MazeEnvironment(
            size=args.size,
            mode=args.mode,
            seed=args.seed,
            hint_style=args.hint_style,
        )
    except ValueError as exc:
        print(f"a_maze: {exc}", file=sys.stderr)
        return 2
    return serve(env, visualize=args.visualize)


if __name__ == "__main__":
    raise SystemExit(main())
