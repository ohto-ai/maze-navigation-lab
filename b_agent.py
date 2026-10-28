#!/usr/bin/env python3
"""B-side exploring agent.

The agent starts an A process and talks to it over stdin/stdout JSON lines.
It never calls A's ``state`` command and never receives absolute coordinates.
Its only coordinate system is relative to its own starting cell.

Feedback modes:
    none      -- success/is_goal only           strategy S0 (DFS + backtrack)
    distance  -- Manhattan distance to goal      strategy S1 (greedy frontiers)
    direction -- compass direction to goal       strategy S2 (hint frontiers)
    shortest  -- true shortest-path distance     strategy S3 (oracle gradient)

``shortest`` is an optional leaky/oracle control, not one of the three fair
modes from the original brief.  It lets us observe the ceiling enabled by a
perfect gradient.


The process's stdout contains exactly one final JSON report.  Diagnostics and
optional visualization go to stderr.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

FREE = 0
WALL = 1

# The relative coordinate frame is fixed by these absolute directions.
DIR_ITEMS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("up", (-1, 0)),
    ("right", (0, 1)),
    ("down", (1, 0)),
    ("left", (0, -1)),
)
DIR_VEC = dict(DIR_ITEMS)
DIR_ORDER = [name for name, _ in DIR_ITEMS]
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
HINTS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
HINT_PREF = {
    "N": ["up", "left", "right", "down"],
    "NE": ["up", "right", "left", "down"],
    "E": ["right", "up", "down", "left"],
    "SE": ["down", "right", "left", "up"],
    "S": ["down", "left", "right", "up"],
    "SW": ["down", "left", "right", "up"],
    "W": ["left", "up", "down", "right"],
    "NW": ["up", "left", "right", "down"],
}


class AgentInternalError(RuntimeError):
    """The maze violated the determinism/feedback contract."""


class StepLimitError(RuntimeError):
    """The configured movement budget has been exhausted."""


class TransportError(RuntimeError):
    """Communication with A failed after all retries."""


class ResponseError(ValueError):
    """A response is syntactically invalid or violates the protocol."""


def add(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIR_VEC[direction]
    return cell[0] + dr, cell[1] + dc


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Transport


class AClient:
    """Line-oriented JSON transport with bounded retries and request ids.

    Request ids are a backward-compatible extension: A echoes them and caches
    the last 1024 responses.  A retried request therefore cannot double-move
    the car if the original response merely arrived late.
    """

    def __init__(
        self,
        command: list[str],
        *,
        mode: str,
        timeout: float = 5.0,
        retries: int = 3,
    ) -> None:
        self.mode = mode
        self.timeout = timeout
        self.max_attempts = retries + 1
        self._next_id = 0
        self.attempts = 0
        self.retries_used = 0
        self.timeouts = 0
        self.protocol_errors = 0

        if not command:
            raise TransportError("A command is empty")
        try:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise TransportError(f"cannot start A: {exc}") from exc

        self._queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._reader = threading.Thread(
            target=self._reader_loop, name="a-reader", daemon=True
        )
        self._reader.start()

    def _reader_loop(self) -> None:
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self._queue.put(("line", line))
        except Exception:  # noqa: BLE001 - report EOF below in all cases
            pass
        finally:
            self._queue.put(("eof", None))

    def _send(self, payload: dict[str, Any]) -> None:
        if self.proc.poll() is not None and self.proc.returncode is not None:
            raise TransportError(f"A exited with code {self.proc.returncode}")
        if self.proc.stdin is None:
            raise TransportError("A stdin is unavailable")
        data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        try:
            self.proc.stdin.write(data + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise TransportError(f"cannot write to A: {exc}") from exc

    def _wait_for(self, req_id: int, deadline: float) -> dict[str, Any]:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for A")
            try:
                kind, payload = self._queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("timed out waiting for A") from exc

            if kind == "eof":
                raise TransportError("A closed its stdout")
            assert payload is not None
            line = payload.strip()
            if not line:
                raise ResponseError("empty response line")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResponseError(f"invalid JSON response: {exc}") from exc
            if not isinstance(response, dict):
                raise ResponseError("response is not a JSON object")

            if "error" in response:
                raise ResponseError(f"A error response: {response['error']}")

            response_id = response.get("id")
            if (
                response_id is not None
                and req_id is not None
                and response_id != req_id
            ):
                # A delayed response to an earlier attempt: discard it.
                continue
            return response

    def _validate_move_response(self, response: dict[str, Any]) -> None:
        success = response.get("success")
        is_goal = response.get("is_goal")
        if type(success) is not bool or type(is_goal) is not bool:
            raise ResponseError("move response needs boolean success/is_goal")

        if self.mode == "distance":
            distance = response.get("distance")
            if type(distance) is not int or distance < 0:
                raise ResponseError(
                    "distance mode requires a non-negative integer distance"
                )
        elif self.mode == "shortest":
            distance = response.get("shortest_distance")
            if type(distance) is not int or distance < 0:
                raise ResponseError(
                    "shortest mode requires a non-negative integer "
                    "shortest_distance"
                )
        elif self.mode == "direction":
            hint = response.get("dir_hint")
            valid_hint = isinstance(hint, str) and hint in HINTS
            if is_goal:
                # At the goal A reports null; either null or a valid hint is ok.
                if hint is not None and not valid_hint:
                    raise ResponseError("invalid dir_hint")
            elif not valid_hint:
                raise ResponseError("direction mode requires a valid dir_hint")

    def move(self, direction: str) -> dict[str, Any]:
        if direction not in DIR_VEC:
            raise ValueError(f"invalid direction {direction!r}")
        self._next_id += 1
        req_id = self._next_id
        payload = {"cmd": "move", "dir": direction, "id": req_id}
        last_error: Exception | None = None

        for attempt in range(self.max_attempts):
            self.attempts += 1
            if attempt:
                self.retries_used += 1
            try:
                self._send(payload)
                deadline = time.monotonic() + self.timeout
                response = self._wait_for(req_id, deadline)
                self._validate_move_response(response)
                return response
            except TimeoutError as exc:
                self.timeouts += 1
                last_error = exc
            except ResponseError as exc:
                self.protocol_errors += 1
                last_error = exc
            except TransportError as exc:
                last_error = exc

            if attempt + 1 < self.max_attempts:
                time.sleep(0.01)

        raise TransportError(
            f"move {direction!r} failed after {self.max_attempts} attempts: "
            f"{last_error}"
        )

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.write(
                        json.dumps({"cmd": "quit"}, separators=(",", ":")) + "\n"
                    )
                    self.proc.stdin.flush()
                    self.proc.stdin.close()
                self.proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001 - cleanup must never mask results
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=1.0)
        if self.proc.stdout is not None:
            try:
                self.proc.stdout.close()
            except OSError:
                pass
        self._reader.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Relative-world agent


class Navigator:
    """Relative map + movement bookkeeping.

    ``map_data`` cells are coordinate offsets from the initial cell:
    0 = known free, 1 = known wall, absent = unknown.  The absolute start is
    never known by this process.
    """

    def __init__(self, client: AClient, mode: str, max_steps: int) -> None:
        self.client = client
        self.mode = mode
        self.max_steps = max_steps
        self.pos = (0, 0)
        self.map_data: dict[tuple[int, int], int] = {(0, 0): FREE}
        self.distances: dict[tuple[int, int], int] = {}
        self.hints: dict[tuple[int, int], str] = {}
        self.steps = 0
        self.wall_hits = 0
        self.reached = False
        self.trace: list[tuple[int, int]] = [(0, 0)]

    @property
    def explored_cells(self) -> int:
        return sum(1 for value in self.map_data.values() if value == FREE)

    def unknown_dirs(self, cell: tuple[int, int] | None = None) -> list[str]:
        origin = self.pos if cell is None else cell
        return [
            direction
            for direction, _ in DIR_ITEMS
            if add(origin, direction) not in self.map_data
        ]

    def step(self, direction: str) -> dict[str, Any]:
        if self.reached:
            raise AgentInternalError("attempted to move after reaching the goal")
        if self.steps >= self.max_steps:
            raise StepLimitError(f"max steps ({self.max_steps}) reached")

        response = self.client.move(direction)
        target = add(self.pos, direction)
        prior = self.map_data.get(target)
        success = response["success"]

        # Update only after a valid response; transport failures never reach
        # this point, so they cannot contaminate the map.
        self.steps += 1
        if success:
            if prior == WALL:
                raise AgentInternalError(
                    f"determinism violation: {target} was a wall but move succeeded"
                )
            self.pos = target
            self.map_data[target] = FREE
            self.trace.append(target)
        else:
            if prior == FREE:
                raise AgentInternalError(
                    f"determinism violation: {target} was free but move failed"
                )
            self.map_data[target] = WALL
            self.wall_hits += 1

        if self.mode in {"distance", "shortest"}:
            field = "distance" if self.mode == "distance" else "shortest_distance"
            distance = response[field]
            old = self.distances.get(self.pos)
            if old is not None and old != distance:
                raise AgentInternalError(
                    "determinism violation: distance changed without movement"
                )
            self.distances[self.pos] = distance
        elif self.mode == "direction":
            hint = response.get("dir_hint")
            if hint is not None:
                self.hints[self.pos] = hint

        if response["is_goal"]:
            self.reached = True
        return response

    def route_to(self, target: tuple[int, int]) -> list[str] | None:
        """Shortest move sequence through known free cells (BFS)."""
        if target == self.pos:
            return []
        previous: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen = {self.pos}
        q: deque[tuple[int, int]] = deque([self.pos])
        while q:
            cell = q.popleft()
            for direction in DIR_ORDER:
                nb = add(cell, direction)
                if nb == target:
                    previous[nb] = (cell, direction)
                    path: list[str] = []
                    cur = nb
                    while cur != self.pos:
                        parent, step_dir = previous[cur]
                        path.append(step_dir)
                        cur = parent
                    path.reverse()
                    return path
                if nb not in seen and self.map_data.get(nb) == FREE:
                    seen.add(nb)
                    previous[nb] = (cell, direction)
                    q.append(nb)
        return None

    def frontier_cells(self) -> list[tuple[int, int]]:
        return [
            cell
            for cell, value in self.map_data.items()
            if value == FREE and self.unknown_dirs(cell)
        ]

    def travel_distances(self) -> dict[tuple[int, int], int]:
        dist = {self.pos: 0}
        q: deque[tuple[int, int]] = deque([self.pos])
        while q:
            cell = q.popleft()
            for direction in DIR_ORDER:
                nb = add(cell, direction)
                if nb not in dist and self.map_data.get(nb) == FREE:
                    dist[nb] = dist[cell] + 1
                    q.append(nb)
        return dist

    def estimated_goal_from_hints(self) -> tuple[int, int]:
        """Intersect all observed sign constraints into a representative goal.

        Hints give only signs/equalities, never magnitude.  The representative
        is the exact coordinate when both axes are pinned, the midpoint of a
        bounded interval, or the known boundary for a half-infinite interval.
        """
        row_lo: int | None = None
        row_hi: int | None = None
        col_lo: int | None = None
        col_hi: int | None = None
        row_exact: int | None = None
        col_exact: int | None = None

        def lower(old: int | None, value: int) -> int:
            return value if old is None else max(old, value)

        def upper(old: int | None, value: int) -> int:
            return value if old is None else min(old, value)

        for (r, c), hint in self.hints.items():
            if hint == "N":
                row_hi = upper(row_hi, r - 1)
                col_exact = c if col_exact is None else col_exact
            elif hint == "NE":
                row_hi = upper(row_hi, r - 1)
                col_lo = lower(col_lo, c + 1)
            elif hint == "E":
                row_exact = r if row_exact is None else row_exact
                col_lo = lower(col_lo, c + 1)
            elif hint == "SE":
                row_lo = lower(row_lo, r + 1)
                col_lo = lower(col_lo, c + 1)
            elif hint == "S":
                row_lo = lower(row_lo, r + 1)
                col_exact = c if col_exact is None else col_exact
            elif hint == "SW":
                row_lo = lower(row_lo, r + 1)
                col_hi = upper(col_hi, c - 1)
            elif hint == "W":
                row_exact = r if row_exact is None else row_exact
                col_hi = upper(col_hi, c - 1)
            elif hint == "NW":
                row_hi = upper(row_hi, r - 1)
                col_hi = upper(col_hi, c - 1)

        def representative(
            lo: int | None, hi: int | None, exact: int | None
        ) -> int:
            if exact is not None:
                return exact
            if lo is not None and hi is not None:
                return (lo + hi) // 2
            if lo is not None:
                return lo
            if hi is not None:
                return hi
            return 0

        return (
            representative(row_lo, row_hi, row_exact),
            representative(col_lo, col_hi, col_exact),
        )

    def render_relative(self) -> str:
        """Relative-only rendering for diagnostics; no A coordinates."""
        cells = list(self.map_data) + [self.pos]
        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)
        trace = set(self.trace)
        lines = [
            f"relative map (origin (0,0), car at {self.pos}, "
            f"{self.explored_cells} explored cells)"
        ]
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                cell = (r, c)
                if cell == self.pos:
                    row.append("C")
                elif cell == (0, 0):
                    row.append("S")
                elif self.map_data.get(cell) == WALL:
                    row.append("#")
                elif self.map_data.get(cell) == FREE:
                    row.append("*" if cell in trace else ".")
                else:
                    row.append("?")
            lines.append("".join(row))
        lines.append("legend: S=start C=car *=walked .=known free #=wall ?=unknown")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategies


class BaseStrategy:
    name = "base"

    def __init__(self, agent: Navigator) -> None:
        self.agent = agent

    def run(self) -> None:
        raise NotImplementedError

    def _calibrate_start(self) -> None:
        """Learn the start cell's distance/hint with at most two moves.

        A only reports feedback after a move.  One wall probe or one
        move-and-return reveals the start observation without leaving the car
        away from the start.  (A goal found during this probe ends the run.)
        """
        agent = self.agent
        if agent.mode in {"distance", "shortest"}:
            need = agent.pos not in agent.distances
        elif agent.mode == "direction":
            need = agent.pos not in agent.hints
        else:
            need = False
        if not need:
            return

        for direction in DIR_ORDER:
            if agent.reached or agent.steps >= agent.max_steps:
                return
            nb = add(agent.pos, direction)
            if nb in agent.map_data:
                continue
            response = agent.step(direction)
            if agent.reached:
                return
            if response["success"]:
                # Return so the start cell also receives a feedback sample.
                if agent.steps >= agent.max_steps:
                    return
                back = agent.step(OPPOSITE[direction])
                if not back["success"]:
                    raise AgentInternalError("reverse move failed in a deterministic maze")
            return


class DFSStrategy(BaseStrategy):
    """S0: fixed-order DFS over the unknown relative map, with backtracking."""

    name = "s0"

    def run(self) -> None:
        agent = self.agent
        # Each entry is (parent_pos, direction taken).  Successful descents
        # form the current root-to-car path.
        stack: list[tuple[tuple[int, int], str]] = []

        while not agent.reached and agent.steps < agent.max_steps:
            advanced = False
            for direction in DIR_ORDER:
                if agent.reached or agent.steps >= agent.max_steps:
                    break
                nb = add(agent.pos, direction)
                if nb in agent.map_data:
                    continue  # already known free or wall

                parent = agent.pos
                response = agent.step(direction)
                if agent.reached:
                    break
                if response["success"]:
                    stack.append((parent, direction))
                    advanced = True
                    break
                # A wall is now recorded; try the next unknown direction.

            if agent.reached:
                break
            if advanced:
                continue
            if not stack:
                break  # all cells reachable from the start have been visited
            parent, direction = stack.pop()
            if agent.pos != add(parent, direction):
                raise AgentInternalError("DFS stack does not match current position")
            back = agent.step(OPPOSITE[direction])
            if not back["success"]:
                raise AgentInternalError("DFS backtrack hit an unexpected wall")

        if not agent.reached and agent.steps < agent.max_steps:
            raise AgentInternalError(
                "DFS exhausted the reachable component without reaching the goal"
            )


class FrontierStrategy(BaseStrategy):
    """S1/S2: greedy frontier exploration with known-map route planning.

    S1 ranks a frontier cell by its observed Manhattan distance to the goal
    plus travel cost.  S2 ranks by distance to a relative goal estimate built
    from the sign constraints in direction hints.
    """

    def __init__(self, agent: Navigator) -> None:
        super().__init__(agent)
        self.name = "s1" if agent.mode == "distance" else "s2"

    def _heuristic(self, cell: tuple[int, int]) -> int:
        agent = self.agent
        if agent.mode == "distance":
            try:
                return agent.distances[cell]
            except KeyError as exc:
                raise AgentInternalError(
                    f"missing distance observation for {cell}"
                ) from exc
        goal = agent.estimated_goal_from_hints()
        return manhattan(cell, goal)

    def _choose_target(
        self, travel: dict[tuple[int, int], int]
    ) -> tuple[tuple[int, int], int] | None:
        candidates = [
            cell for cell in self.agent.frontier_cells() if cell in travel
        ]
        if not candidates:
            return None
        scored = []
        for cell in candidates:
            heuristic = self._heuristic(cell)
            scored.append(
                (
                    heuristic + travel[cell],
                    heuristic,
                    travel[cell],
                    cell,
                )
            )
        _, _, travel_cost, target = min(scored)
        return target, travel_cost

    def _direction_order(self, cell: tuple[int, int]) -> list[str]:
        unknown = self.agent.unknown_dirs(cell)
        if self.agent.mode == "direction":
            goal = self.agent.estimated_goal_from_hints()
            hint = self.agent.hints.get(cell)
            preferred = HINT_PREF.get(hint or "", DIR_ORDER)
            rank = {direction: i for i, direction in enumerate(preferred)}
            return sorted(
                unknown,
                key=lambda direction: (
                    manhattan(add(cell, direction), goal),
                    rank.get(direction, len(rank)),
                    DIR_ORDER.index(direction),
                ),
            )
        return [direction for direction in DIR_ORDER if direction in unknown]

    def _navigate(self, target: tuple[int, int]) -> None:
        agent = self.agent
        route = agent.route_to(target)
        if route is None:
            raise AgentInternalError(f"no known route to frontier {target}")
        for direction in route:
            if agent.reached:
                return
            response = agent.step(direction)
            if not response["success"]:
                raise AgentInternalError("move through known free cell failed")

    def run(self) -> None:
        agent = self.agent
        self._calibrate_start()
        if agent.reached:
            return

        while not agent.reached and agent.steps < agent.max_steps:
            travel = agent.travel_distances()
            choice = self._choose_target(travel)
            if choice is None:
                break
            target, _ = choice
            if target != agent.pos:
                self._navigate(target)
                if agent.reached:
                    break

            unknowns = agent.unknown_dirs(agent.pos)
            if not unknowns:
                continue
            ordered = self._direction_order(agent.pos)
            direction = ordered[0]
            agent.step(direction)
            # The loop re-ranks all frontiers after learning free/wall/goal.
        # End


class GradientStrategy(BaseStrategy):
    """S3: oracle gradient descent using true shortest-path distances.

    This is an intentionally leaky control mode, not one of the three fair
    strategies.  At a cell with true distance ``d``, a free neighbour is either
    at ``d-1`` (the unique corridor towards the goal), ``d+1`` (away), or is a
    wall.  Wrong free branches are stepped into once and immediately abandoned;
    every discovered wall is remembered.  Thus the agent reaches the goal with
    only local downhill moves and never needs whole-map exploration.
    """

    name = "s3"

    def __init__(self, agent: Navigator) -> None:
        super().__init__(agent)
        self._last_dir: str | None = None

    def _known_lower(self, distance: int) -> list[str]:
        agent = self.agent
        lower = []
        for direction in DIR_ORDER:
            nb = add(agent.pos, direction)
            if agent.map_data.get(nb) != FREE:
                continue
            if agent.distances.get(nb) == distance - 1:
                lower.append(direction)
        if self._last_dir in lower:
            lower.remove(self._last_dir)
            lower.insert(0, self._last_dir)
        return lower

    def _probe_order(self, unknown: list[str]) -> list[str]:
        ordered = []
        if self._last_dir in unknown:
            ordered.append(self._last_dir)
        for direction in DIR_ORDER:
            if direction in unknown and direction not in ordered:
                ordered.append(direction)
        return ordered

    def run(self) -> None:
        agent = self.agent
        self._calibrate_start()
        if agent.reached:
            return

        while not agent.reached and agent.steps < agent.max_steps:
            distance = agent.distances.get(agent.pos)
            if distance is None:
                raise AgentInternalError("missing shortest distance for current cell")

            known_lower = self._known_lower(distance)
            if known_lower:
                direction = known_lower[0]
                response = agent.step(direction)
                if not response["success"]:
                    raise AgentInternalError(
                        "gradient move along known-free cell failed"
                    )
                self._last_dir = direction
                continue

            unknown = agent.unknown_dirs(agent.pos)
            if not unknown:
                raise AgentInternalError(
                    "shortest-distance gradient has no lower neighbour "
                    f"at {agent.pos}"
                )

            moved = False
            for direction in self._probe_order(unknown):
                if agent.reached or agent.steps >= agent.max_steps:
                    break
                response = agent.step(direction)
                if agent.reached:
                    break
                if not response["success"]:
                    continue

                neighbour_distance = agent.distances.get(agent.pos)
                if neighbour_distance is None:
                    raise AgentInternalError(
                        "successful move produced no shortest-distance feedback"
                    )
                if neighbour_distance == distance - 1:
                    self._last_dir = direction
                    moved = True
                    break
                if neighbour_distance == distance + 1:
                    # Wrong branch: undo immediately.  Its wall/away status is
                    # now cached, so it will not be tried again.
                    back = agent.step(OPPOSITE[direction])
                    if agent.reached:
                        break
                    if not back["success"]:
                        raise AgentInternalError(
                            "cannot backtrack from non-gradient branch"
                        )
                    continue
                raise AgentInternalError(
                    "shortest distance changed by more than one step"
                )

            if agent.reached:
                break
            if not moved:
                if agent.steps >= agent.max_steps:
                    break
                raise AgentInternalError(
                    "gradient search exhausted without a descending move"
                )

        if not agent.reached and agent.steps < agent.max_steps:
            raise AgentInternalError(
                "shortest-distance gradient ended before reaching the goal"
            )


# ---------------------------------------------------------------------------
# Entry point


def default_a_command(mode: str) -> list[str]:
    a_path = Path(__file__).with_name("a_maze.py")
    return [
        sys.executable,
        "-u",
        str(a_path),
        "--size",
        "10",
        "--mode",
        mode,
        "--seed",
        "42",
    ]


def parse_a_command(text: str | None, mode: str) -> list[str]:
    if text is None:
        return default_a_command(mode)
    try:
        command = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--a-command must be a JSON string array: {exc}") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) for part in command)
    ):
        raise ValueError("--a-command must be a non-empty JSON array of strings")
    return command


def resolve_strategy(mode: str, requested: str) -> str:
    if requested == "auto":
        return {
            "none": "s0",
            "distance": "s1",
            "direction": "s2",
            "shortest": "s3",
        }[mode]
    if requested == "s1" and mode != "distance":
        raise ValueError("strategy s1 requires --mode distance")
    if requested == "s2" and mode != "direction":
        raise ValueError("strategy s2 requires --mode direction")
    if requested == "s3" and mode != "shortest":
        raise ValueError("strategy s3 requires --mode shortest")
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="B-side relative-navigation agent (JSON lines to A)"
    )
    parser.add_argument(
        "--mode",
        choices=("none", "distance", "direction", "shortest"),
        required=True,
    )
    parser.add_argument(
        "--strategy",
        choices=("auto", "s0", "s1", "s2", "s3"),
        default="auto",
        help="auto maps none/distance/direction/shortest to s0/s1/s2/s3",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--retries", type=int, default=3, help="retries after the first attempt"
    )
    parser.add_argument(
        "--a-command",
        default=None,
        help="JSON array launching A; defaults to this directory's a_maze.py",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="print B's relative map to stderr after the run",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the stderr summary"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.perf_counter()

    report: dict[str, Any] = {
        "mode": args.mode,
        "strategy": args.strategy,
        "steps": 0,
        "reached": False,
        "explored_cells": 0,
        "time": 0.0,
    }
    client: AClient | None = None
    agent: Navigator | None = None
    exit_code = 0

    try:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        if args.retries < 0 or args.timeout <= 0:
            raise ValueError("--retries must be >= 0 and --timeout must be > 0")
        strategy_name = resolve_strategy(args.mode, args.strategy)
        report["strategy"] = strategy_name
        command = parse_a_command(args.a_command, args.mode)

        client = AClient(
            command, mode=args.mode, timeout=args.timeout, retries=args.retries
        )
        agent = Navigator(client, mode=args.mode, max_steps=args.max_steps)
        strategy_map = {
            "s0": DFSStrategy,
            "s1": FrontierStrategy,
            "s2": FrontierStrategy,
            "s3": GradientStrategy,
        }
        strategy = strategy_map[strategy_name](agent)

        try:
            strategy.run()
        except StepLimitError:
            report["reason"] = "step_limit"
            exit_code = 4
        except AgentInternalError as exc:
            report["reason"] = f"internal_error: {exc}"
            exit_code = 3
    except (ValueError, TransportError) as exc:
        report["error"] = str(exc)
        exit_code = 2
    finally:
        elapsed = time.perf_counter() - started
        if agent is not None:
            report["steps"] = agent.steps
            report["reached"] = agent.reached
            report["explored_cells"] = agent.explored_cells
            report["wall_hits"] = agent.wall_hits
        if client is not None:
            report["communication_retries"] = client.retries_used
            report["communication_timeouts"] = client.timeouts
            report["protocol_errors_seen"] = client.protocol_errors
            client.close()
        report["time"] = round(elapsed, 6)
        if agent is not None and not report["reached"]:
            if "reason" not in report and "error" not in report:
                report["reason"] = (
                    "step_limit"
                    if agent.steps >= agent.max_steps
                    else "search_exhausted"
                )
            if exit_code == 0:
                exit_code = 1

        print(json.dumps(report, ensure_ascii=True, separators=(",", ":")))
        if args.visualize and agent is not None:
            print(agent.render_relative(), file=sys.stderr, flush=True)
        elif not args.quiet and agent is not None:
            print(
                f"B: strategy={report['strategy']} steps={report['steps']} "
                f"reached={report['reached']} explored={report['explored_cells']}",
                file=sys.stderr,
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
