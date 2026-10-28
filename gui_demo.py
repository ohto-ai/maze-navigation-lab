#!/usr/bin/env python3
"""Interactive Tkinter visualization for the maze A/B project.

The demo replays real A/B strategy runs in-process.  It runs the same
``MazeEnvironment`` and strategy classes as the CLI programs, records one
snapshot per movement, then animates either the three fair feedback modes side
by side or the shortest-distance gradient oracle.

The gradient view draws every cell's true shortest distance plus downhill
arrows, making the otherwise hidden gradient directly inspectable.  The drawing
layer is deliberately omniscient (it can see the whole maze and the goal), while
the B strategies still make every decision from the allowed relative feedback.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from typing import Any

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox
except ImportError as exc:  # pragma: no cover - Windows Python ships Tk
    print(
        "gui_demo requires tkinter (normally included with Python).",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

from a_maze import MazeEnvironment, WALL, shortest_distances
from b_agent import (
    DIR_ITEMS,
    DIR_VEC,
    DFSStrategy,
    FrontierStrategy,
    GradientStrategy,
    Navigator,
)


# ---------------------------------------------------------------------------
# Design tokens (validated dark-mode data palette)

PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
SURFACE_RAISED = "#232321"
SURFACE_DEEP = "#141412"
BORDER = "#383835"
PRIMARY = "#f0efec"
SECONDARY = "#c3c2b7"
MUTED = "#898781"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
GOAL = "#d55181"

MODE_BLUE = "#3987e5"
MODE_ORANGE = "#d95926"
MODE_AQUA = "#199e70"
MODE_YELLOW = "#c98500"
GRADIENT_BLUE = "#3987e5"

WALL_UNKNOWN = "#272725"
PASSAGE_UNKNOWN = "#10100f"
KNOWN_WALL = "#332c29"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def blend(top: str, bottom: str, alpha: float) -> str:
    """Opaque approximation of alpha compositing for Tk fills."""
    alpha = max(0.0, min(1.0, alpha))
    tr, tg, tb = _rgb(top)
    br, bg, bb = _rgb(bottom)
    return "#{:02x}{:02x}{:02x}".format(
        round(tr * alpha + br * (1 - alpha)),
        round(tg * alpha + bg * (1 - alpha)),
        round(tb * alpha + bb * (1 - alpha)),
    )


@dataclass(frozen=True)
class ModeSpec:
    key: str
    title: str
    tag: str
    subtitle: str
    color: str


MODE_SPECS: tuple[ModeSpec, ...] = (
    ModeSpec(
        key="none",
        title="Binary feedback",
        tag="none · S0",
        subtitle="DFS + backtracking · success / bump only",
        color=MODE_BLUE,
    ),
    ModeSpec(
        key="distance",
        title="Manhattan distance",
        tag="distance · S1",
        subtitle="Greedy frontier + backtracking · L1 distance",
        color=MODE_ORANGE,
    ),
    ModeSpec(
        key="direction",
        title="Compass direction",
        tag="direction · S2",
        subtitle="Goal-quadrant estimate + backtracking · 8-way hint",
        color=MODE_AQUA,
    ),
)

SHORTEST_SPEC = ModeSpec(
    key="shortest",
    title="Shortest-path gradient",
    tag="shortest · S3",
    subtitle="True distance · numbers/arrows step downhill (oracle)",
    color=MODE_YELLOW,
)
ALL_SPECS: tuple[ModeSpec, ...] = MODE_SPECS + (SHORTEST_SPEC,)


# ---------------------------------------------------------------------------
# Recorded traces


@dataclass(frozen=True)
class FrameSnapshot:
    step: int
    car: tuple[int, int]
    known: dict[tuple[int, int], int]
    trail: tuple[tuple[int, int], ...]
    attempted: tuple[int, int] | None
    success: bool | None
    is_goal: bool
    distance: int | None
    shortest_distance: int | None
    hint: str | None
    explored: int
    wall_hits: int


@dataclass(frozen=True)
class Trace:
    spec: ModeSpec
    maze: tuple[tuple[int, ...], ...]
    start: tuple[int, int]
    goal: tuple[int, int]
    frames: tuple[FrameSnapshot, ...]
    gradient: dict[tuple[int, int], int]
    downhill: dict[tuple[int, int], str]

    @property
    def final_steps(self) -> int:
        return self.frames[-1].step

    @property
    def reached(self) -> bool:
        return self.frames[-1].is_goal


class _LocalClient:
    def __init__(self, env: MazeEnvironment) -> None:
        self.env = env

    def move(self, direction: str) -> dict[str, Any]:
        return self.env.move(direction)


class _RecordingNavigator(Navigator):
    """Navigator that snapshots the view after every movement.

    Only the demo layer sees A's absolute state; strategy decisions are made by
    the unmodified Navigator/strategy code from b_agent.py.
    """

    def __init__(self, env: MazeEnvironment, mode: str, max_steps: int) -> None:
        self.env = env
        self.origin = env.start
        super().__init__(_LocalClient(env), mode=mode, max_steps=max_steps)
        self.frames: list[FrameSnapshot] = []
        self._record(None, None)

    def _to_absolute(
        self, cell: tuple[int, int]
    ) -> tuple[int, int]:
        return self.origin[0] + cell[0], self.origin[1] + cell[1]

    def _record(
        self,
        response: dict[str, Any] | None,
        attempted: tuple[int, int] | None,
    ) -> None:
        known = {
            self._to_absolute(cell): value
            for cell, value in self.map_data.items()
        }
        self.frames.append(
            FrameSnapshot(
                step=self.steps,
                car=self.env.car,
                known=known,
                trail=tuple(self._to_absolute(cell) for cell in self.trace),
                attempted=attempted,
                success=None if response is None else response["success"],
                is_goal=self.env.car == self.env.goal,
                distance=(
                    None if response is None else response.get("distance")
                ),
                shortest_distance=(
                    None
                    if response is None
                    else response.get("shortest_distance")
                ),
                hint=None if response is None else response.get("dir_hint"),
                explored=self.explored_cells,
                wall_hits=self.wall_hits,
            )
        )

    def step(self, direction: str) -> dict[str, Any]:
        dr, dc = DIR_VEC[direction]
        attempted = (self.env.car[0] + dr, self.env.car[1] + dc)
        response = super().step(direction)
        self._record(response, attempted)
        return response


def _downhill_map(
    gradient: dict[tuple[int, int], int], maze: tuple[tuple[int, ...], ...]
) -> dict[tuple[int, int], str]:
    """For each cell, the unique direction that reduces shortest distance."""
    downhill: dict[tuple[int, int], str] = {}
    for cell, distance in gradient.items():
        if distance <= 0:
            continue
        r, c = cell
        for direction, (dr, dc) in DIR_ITEMS:
            nb = (r + dr, c + dc)
            if gradient.get(nb) == distance - 1:
                downhill[cell] = direction
                break
    return downhill


def build_traces(size: int, seed: int) -> dict[str, Trace]:
    """Run the fair modes plus the shortest-gradient oracle on one maze."""
    strategies = {
        "none": DFSStrategy,
        "distance": FrontierStrategy,
        "direction": FrontierStrategy,
        "shortest": GradientStrategy,
    }
    traces: dict[str, Trace] = {}
    reference: tuple[Any, Any, Any] | None = None
    gradient: dict[tuple[int, int], int] = {}
    downhill: dict[tuple[int, int], str] = {}

    for spec in ALL_SPECS:
        env = MazeEnvironment(size=size, mode=spec.key, seed=seed)
        maze = tuple(tuple(row) for row in env.maze)
        layout = (maze, env.start, env.goal)
        if reference is None:
            reference = layout
            gradient = shortest_distances(
                [list(row) for row in maze], env.goal
            )
            downhill = _downhill_map(gradient, maze)
        elif layout != reference:
            raise RuntimeError("modes generated different mazes for the same seed")

        navigator = _RecordingNavigator(env, spec.key, max_steps=200_000)
        strategy = strategies[spec.key](navigator)
        strategy.run()
        traces[spec.key] = Trace(
            spec=spec,
            maze=maze,
            start=env.start,
            goal=env.goal,
            frames=tuple(navigator.frames),
            gradient=gradient,
            downhill=downhill,
        )

    return traces


# ---------------------------------------------------------------------------
# Visual panel


class MazePanel:
    def __init__(
        self,
        parent: tk.Misc,
        spec: ModeSpec,
        *,
        fonts: dict[str, tuple],
        show_gradient: bool = False,
    ) -> None:
        self.spec = spec
        self.fonts = fonts
        self.show_gradient = show_gradient
        self.trace: Trace | None = None
        self.current: FrameSnapshot | None = None
        self.local_index = -1
        self.max_gradient = 1
        self._last_draw_size: tuple[int, int] | None = None

        self.known_free_fill = blend(spec.color, SURFACE, 0.22)
        self.card = tk.Frame(
            parent,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
            bd=0,
        )
        self.card.grid_rowconfigure(2, weight=1)
        self.card.grid_columnconfigure(0, weight=1)

        accent = tk.Frame(self.card, bg=spec.color, height=3, bd=0)
        accent.grid(row=0, column=0, sticky="ew")

        header = tk.Frame(self.card, bg=SURFACE)
        header.grid(row=1, column=0, sticky="ew", padx=14, pady=(11, 0))
        header.grid_columnconfigure(0, weight=1)

        title_line = tk.Frame(header, bg=SURFACE)
        title_line.grid(row=0, column=0, sticky="w")
        tk.Label(
            title_line,
            text="●",
            bg=SURFACE,
            fg=spec.color,
            font=fonts["dot"],
        ).pack(side="left", padx=(0, 5))
        tk.Label(
            title_line,
            text=spec.title,
            bg=SURFACE,
            fg=PRIMARY,
            font=fonts["panel_title"],
        ).pack(side="left")
        tk.Label(
            title_line,
            text=spec.tag,
            bg=SURFACE_RAISED,
            fg=SECONDARY,
            font=fonts["chip"],
            padx=7,
            pady=2,
        ).pack(side="left", padx=(8, 0))

        status = tk.Frame(header, bg=SURFACE_RAISED)
        status.grid(row=0, column=1, sticky="e")
        self.status_dot = tk.Label(
            status,
            text="●",
            bg=SURFACE_RAISED,
            fg=MUTED,
            font=fonts["status_dot"],
        )
        self.status_dot.pack(side="left", padx=(8, 3), pady=3)
        self.status_label = tk.Label(
            status,
            text="Idle",
            bg=SURFACE_RAISED,
            fg=SECONDARY,
            font=fonts["chip"],
        )
        self.status_label.pack(side="left", padx=(0, 8), pady=3)

        tk.Label(
            header,
            text=spec.subtitle,
            bg=SURFACE,
            fg=MUTED,
            font=fonts["small"],
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        self.canvas = tk.Canvas(
            self.card,
            bg=SURFACE,
            bd=0,
            highlightthickness=0,
        )
        self.canvas.grid(
            row=2, column=0, sticky="nsew", padx=12, pady=(9, 4)
        )
        self.canvas.bind("<Configure>", self._on_resize)

        feedback = tk.Frame(self.card, bg=SURFACE)
        feedback.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 2))
        self.feedback_icon = tk.Label(
            feedback,
            text="·",
            bg=SURFACE,
            fg=MUTED,
            font=fonts["feedback_icon"],
            width=2,
            anchor="w",
        )
        self.feedback_icon.pack(side="left")
        self.feedback_text = tk.Label(
            feedback,
            text="Waiting for first move",
            bg=SURFACE,
            fg=SECONDARY,
            font=fonts["feedback"],
            anchor="w",
        )
        self.feedback_text.pack(side="left", fill="x", expand=True)

        stats = tk.Frame(self.card, bg=SURFACE)
        stats.grid(row=4, column=0, sticky="ew", padx=14, pady=(5, 13))
        for col in range(3):
            stats.grid_columnconfigure(col, weight=1, uniform="stat")
        self.steps_var = tk.StringVar(value="0")
        self.known_var = tk.StringVar(value="1")
        self.walls_var = tk.StringVar(value="0")
        self._stat_tile(stats, 0, "Steps", self.steps_var)
        self._stat_tile(stats, 1, "Known free", self.known_var)
        self._stat_tile(stats, 2, "Bumps", self.walls_var)

    def _stat_tile(
        self, parent: tk.Frame, column: int, label: str, value: tk.StringVar
    ) -> None:
        holder = tk.Frame(parent, bg=SURFACE)
        holder.grid(row=0, column=column, sticky="w")
        tk.Label(
            holder,
            text=label,
            bg=SURFACE,
            fg=MUTED,
            font=self.fonts["small"],
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            holder,
            textvariable=value,
            bg=SURFACE,
            fg=PRIMARY,
            font=self.fonts["stat_value"],
            anchor="w",
        ).pack(anchor="w", pady=(1, 0))

    def set_trace(self, trace: Trace) -> None:
        self.trace = trace
        self.current = None
        self.local_index = -1
        self.max_gradient = max(trace.gradient.values(), default=1) or 1
        self.set_index(0, force=True)

    def set_index(self, global_index: int, *, force: bool = False) -> None:
        if self.trace is None:
            return
        local = min(max(0, global_index), len(self.trace.frames) - 1)
        if not force and local == self.local_index:
            return
        self.local_index = local
        self.current = self.trace.frames[local]
        frame = self.current

        self.steps_var.set(str(frame.step))
        self.known_var.set(str(frame.explored))
        self.walls_var.set(str(frame.wall_hits))

        final = local == len(self.trace.frames) - 1
        if frame.is_goal:
            self.status_dot.configure(fg=GOOD)
            self.status_label.configure(text="Reached")
        elif final:
            self.status_dot.configure(fg=CRITICAL)
            self.status_label.configure(text="Not reached")
        else:
            self.status_dot.configure(fg=MUTED)
            self.status_label.configure(text="Exploring")

        self._update_feedback(local)
        self.redraw(force=True)

    def _update_feedback(self, local: int) -> None:
        assert self.trace is not None and self.current is not None
        frame = self.current
        previous = self.trace.frames[local - 1] if local > 0 else None

        if frame.success is None:
            icon, color = "·", MUTED
            text = "Waiting for first move"
        else:
            icon = "✓" if frame.success else "✕"
            color = GOOD if frame.success else CRITICAL
            if self.spec.key == "none":
                text = "Moved" if frame.success else "Blocked"
            elif self.spec.key == "distance":
                distance = frame.distance
                if frame.success and previous is not None and (
                    previous.distance is not None
                ):
                    text = f"dist {previous.distance} → {distance}"
                elif frame.success:
                    text = f"dist {distance}"
                else:
                    text = f"Blocked · dist {distance}"
            elif self.spec.key == "shortest":
                distance = frame.shortest_distance
                if frame.success and previous is not None and (
                    previous.shortest_distance is not None
                ):
                    text = (
                        "shortest "
                        f"{previous.shortest_distance} → {distance}"
                    )
                elif frame.success:
                    text = f"shortest {distance}"
                else:
                    text = f"Blocked · shortest {distance}"
            else:
                hint = frame.hint
                text = f"hint {hint}" if hint else "Goal reached"
                if not frame.success:
                    text = f"Blocked · {text}"
            if frame.is_goal and not text.endswith("Goal reached"):
                text += " · goal"

        self.feedback_icon.configure(text=icon, fg=color)
        self.feedback_text.configure(text=text)

    def _gradient_cell_fill(self, distance: int) -> str:
        closeness = 1.0 - distance / self.max_gradient
        alpha = 0.03 + 0.36 * max(0.0, min(1.0, closeness))
        return blend(GRADIENT_BLUE, SURFACE, alpha)

    def _draw_gradient_overlay(
        self,
        canvas: tk.Canvas,
        ox: int,
        oy: int,
        cell: int,
        frame: FrameSnapshot,
    ) -> None:
        assert self.trace is not None
        trace = self.trace
        font = (self.fonts["family"], max(7, int(cell * 0.17)))
        for (r, c), distance in trace.gradient.items():
            known_free = frame.known.get((r, c)) == 0
            cx = ox + c * cell + cell / 2
            cy = oy + r * cell + cell / 2
            direction = trace.downhill.get((r, c))
            if direction is not None and cell >= 12:
                dr, dc = DIR_VEC[direction]
                start = cell * 0.25
                end = cell * 0.43
                color = SECONDARY if known_free else MUTED
                width = 2 if known_free else 1
                sx, sy = cx + dc * start, cy + dr * start
                ex, ey = cx + dc * end, cy + dr * end
                canvas.create_line(
                    sx,
                    sy,
                    ex,
                    ey,
                    fill=color,
                    width=width,
                    capstyle=tk.ROUND,
                )
                head = max(2.5, cell * 0.075)
                # Two short strokes form a lightweight open arrowhead.
                p1 = (
                    ex - dc * head - dr * head,
                    ey - dr * head + dc * head,
                )
                p2 = (
                    ex - dc * head + dr * head,
                    ey - dr * head - dc * head,
                )
                canvas.create_line(
                    p1[0],
                    p1[1],
                    ex,
                    ey,
                    p2[0],
                    p2[1],
                    fill=color,
                    width=width,
                    capstyle=tk.ROUND,
                )
            if cell >= 20:
                canvas.create_text(
                    ox + c * cell + cell * 0.12,
                    oy + r * cell + cell * 0.11,
                    text=str(distance),
                    anchor="nw",
                    fill=PRIMARY if known_free else SECONDARY,
                    font=font,
                )

    def _on_resize(self, _event: tk.Event) -> None:
        self.redraw()

    def redraw(self, *, force: bool = False) -> None:
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        size = (width, height)
        if not force and self._last_draw_size == size:
            return
        self._last_draw_size = size

        canvas = self.canvas
        canvas.delete("all")
        if self.trace is None or self.current is None or width < 20 or height < 20:
            return

        trace = self.trace
        frame = self.current
        n = len(trace.maze)
        margin = 10
        cell = max(
            1,
            min((width - 2 * margin) // n, (height - 2 * margin) // n),
        )
        board = cell * n
        ox = (width - board) // 2
        oy = (height - board) // 2

        canvas.create_rectangle(
            ox - 1,
            oy - 1,
            ox + board + 1,
            oy + board + 1,
            outline=BORDER,
            width=1,
            fill=SURFACE,
        )

        gap = 1 if cell >= 8 else 0
        for r in range(n):
            for c in range(n):
                x0 = ox + c * cell + gap
                y0 = oy + r * cell + gap
                x1 = ox + (c + 1) * cell - gap
                y1 = oy + (r + 1) * cell - gap
                known = frame.known.get((r, c))
                if trace.maze[r][c] == WALL:
                    fill = KNOWN_WALL if known == 1 else WALL_UNKNOWN
                elif self.show_gradient:
                    fill = self._gradient_cell_fill(
                        trace.gradient.get((r, c), self.max_gradient)
                    )
                else:
                    fill = (
                        self.known_free_fill
                        if known == 0
                        else PASSAGE_UNKNOWN
                    )
                canvas.create_rectangle(
                    x0, y0, x1, y1, fill=fill, outline="", width=0
                )

        # Trail through successfully visited cells.
        if len(frame.trail) > 1 and cell >= 6:
            points: list[int] = []
            for r, c in frame.trail:
                points.extend(
                    [
                        ox + c * cell + cell // 2,
                        oy + r * cell + cell // 2,
                    ]
                )
            canvas.create_line(
                *points,
                fill=self.spec.color,
                width=2,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )

        if self.show_gradient:
            self._draw_gradient_overlay(canvas, ox, oy, cell, frame)

        # Start badge: top-left normally; bottom-left in the gradient view so
        # it does not collide with the per-cell distance number.
        if cell >= 13:
            font = (self.fonts["family"], max(7, int(cell * 0.22)))
            sr, sc = trace.start
            sx = ox + sc * cell + cell * 0.20
            sy = (
                oy + (sr + 1) * cell - cell * 0.17
                if self.show_gradient
                else oy + sr * cell + cell * 0.18
            )
            canvas.create_text(
                sx,
                sy,
                text="S",
                anchor="sw" if self.show_gradient else "nw",
                fill=SECONDARY,
                font=font,
            )

        # Goal: a magenta diamond with a 2px surface ring.
        gr, gc = trace.goal
        cx = ox + gc * cell + cell / 2
        cy = oy + gr * cell + cell / 2
        radius = max(3.5, cell * 0.28)
        canvas.create_polygon(
            cx,
            cy - radius - 2,
            cx + radius + 2,
            cy,
            cx,
            cy + radius + 2,
            cx - radius - 2,
            cy,
            fill=SURFACE,
            outline="",
        )
        canvas.create_polygon(
            cx,
            cy - radius,
            cx + radius,
            cy,
            cx,
            cy + radius,
            cx - radius,
            cy,
            fill=GOAL,
            outline="",
        )
        if cell >= 18:
            canvas.create_text(
                ox + (gc + 1) * cell - cell * 0.18,
                oy + gr * cell + cell * 0.18,
                text="G",
                anchor="ne",
                fill=SECONDARY,
                font=(self.fonts["family"], max(7, int(cell * 0.22))),
            )

        # Latest wall bump.
        if frame.attempted is not None and frame.success is False:
            ar, ac = frame.attempted
            if 0 <= ar < n and 0 <= ac < n:
                ax = ox + ac * cell + cell / 2
                ay = oy + ar * cell + cell / 2
                arm = max(3.0, cell * 0.17)
                for width, color in ((4, SURFACE), (2, CRITICAL)):
                    canvas.create_line(
                        ax - arm,
                        ay - arm,
                        ax + arm,
                        ay + arm,
                        fill=color,
                        width=width,
                        capstyle=tk.ROUND,
                    )
                    canvas.create_line(
                        ax - arm,
                        ay + arm,
                        ax + arm,
                        ay - arm,
                        fill=color,
                        width=width,
                        capstyle=tk.ROUND,
                    )

        # Car token: primary ink with a 2px surface ring.
        cr, cc = frame.car
        ccx = ox + cc * cell + cell / 2
        ccy = oy + cr * cell + cell / 2
        car_radius = max(4.0, cell * 0.24)
        canvas.create_oval(
            ccx - car_radius - 2,
            ccy - car_radius - 2,
            ccx + car_radius + 2,
            ccy + car_radius + 2,
            fill=SURFACE,
            outline="",
        )
        canvas.create_oval(
            ccx - car_radius,
            ccy - car_radius,
            ccx + car_radius,
            ccy + car_radius,
            fill=PRIMARY,
            outline="",
        )
        if frame.is_goal:
            halo = car_radius + 5
            canvas.create_oval(
                ccx - halo,
                ccy - halo,
                ccx + halo,
                ccy + halo,
                outline=GOAL,
                width=2,
            )


# ---------------------------------------------------------------------------
# Application


class DemoApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        seed: int = 92,
        size: int = 14,
        speed: int = 14,
        autoplay: bool = True,
    ) -> None:
        self.root = root
        self.fonts = self._make_fonts()
        self.playing = False
        self.after_id: str | None = None
        self.index = 0
        self.max_index = 0
        self.seed = seed
        self.size = size
        self.traces: dict[str, Trace] | None = None

        root.title("Maze Navigation Lab · 2×2 Feedback Demo")
        root.configure(bg=PAGE)
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self._fit_window()

        self.seed_var = tk.StringVar(value=str(seed))
        self.size_var = tk.StringVar(value=str(size))
        self.speed_var = tk.IntVar(value=max(1, min(30, speed)))
        self.status_var = tk.StringVar(value="Ready")

        self._build_header()
        self._build_controls()
        self._build_panels()
        self._build_footer()

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.load_run(seed=seed, size=size, autoplay=autoplay)

    # ---------------------------------------------------------------- chrome
    def _make_fonts(self) -> dict[str, Any]:
        families = set(tkfont.families(self.root))
        default_family = tkfont.nametofont("TkDefaultFont").actual("family")
        family = next(
            (
                candidate
                for candidate in (
                    "Segoe UI",
                    "Arial",
                    "Microsoft YaHei UI",
                    "Microsoft YaHei",
                )
                if candidate in families
            ),
            default_family,
        )
        symbol_family = (
            "Segoe UI Symbol" if "Segoe UI Symbol" in families else family
        )
        return {
            "family": family,
            "title": (family, 23, "bold"),
            "subtitle": (family, 10),
            "panel_title": (family, 15, "bold"),
            "small": (family, 9),
            "chip": (family, 9),
            "status_dot": (family, 9),
            "dot": (family, 9),
            "feedback": (family, 10),
            "feedback_icon": (symbol_family, 10),
            "stat_value": (family, 21, "bold"),
            "button": (family, 10),
            "input": (family, 10),
            "footer": (family, 9),
        }

    def _fit_window(self) -> None:
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(1860, screen_w - 80)
        height = min(1320, screen_h - 100)
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1200, 760)

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=PAGE)
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(20, 8))
        header.grid_columnconfigure(0, weight=1)

        left = tk.Frame(header, bg=PAGE)
        left.grid(row=0, column=0, sticky="w")
        tk.Label(
            left,
            text="Maze Navigation Lab",
            bg=PAGE,
            fg=PRIMARY,
            font=self.fonts["title"],
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            left,
            text=(
                "Same seed, same maze, step-synchronised playback. "
                "B sees only relative feedback; the full maze is shown for the viewer."
            ),
            bg=PAGE,
            fg=SECONDARY,
            font=self.fonts["subtitle"],
            anchor="w",
        ).pack(anchor="w", pady=(3, 0))

        badge = tk.Frame(
            header,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=6,
        )
        badge.grid(row=0, column=1, sticky="e")
        tk.Label(
            badge,
            text="●",
            bg=SURFACE,
            fg=MUTED,
            font=self.fonts["small"],
        ).pack(side="left", padx=(0, 5))
        tk.Label(
            badge,
            text="4 modes · full-map view",
            bg=SURFACE,
            fg=SECONDARY,
            font=self.fonts["small"],
        ).pack(side="left")

    def _build_controls(self) -> None:
        bar = tk.Frame(
            self.root,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=14,
            pady=10,
        )
        bar.grid(row=1, column=0, sticky="ew", padx=24, pady=(4, 10))
        bar.grid_columnconfigure(1, weight=1)

        left = tk.Frame(bar, bg=SURFACE)
        left.grid(row=0, column=0, sticky="w")

        tk.Label(
            left,
            text="Seed",
            bg=SURFACE,
            fg=MUTED,
            font=self.fonts["small"],
        ).pack(side="left", padx=(0, 6))
        self.seed_entry = self._spinbox(
            left, self.seed_var, from_=0, to=999_999, width=7
        )
        self.seed_entry.pack(side="left")
        self.seed_entry.bind("<Return>", lambda _event: self._regenerate())

        tk.Label(
            left,
            text="Size",
            bg=SURFACE,
            fg=MUTED,
            font=self.fonts["small"],
        ).pack(side="left", padx=(13, 6))
        self.size_entry = self._spinbox(
            left, self.size_var, from_=4, to=20, width=4
        )
        self.size_entry.pack(side="left")
        self.size_entry.bind("<Return>", lambda _event: self._regenerate())

        self._button(
            left, "Generate & play", self._regenerate, primary=True
        ).pack(side="left", padx=(14, 7))
        self._button(left, "Random seed", self._randomize).pack(side="left")

        right = tk.Frame(bar, bg=SURFACE)
        right.grid(row=0, column=1, sticky="e")

        self.play_button = self._button(right, "Pause", self.toggle_play)
        self.play_button.pack(side="left", padx=(0, 7))
        self._button(right, "Step", self.step_once).pack(side="left", padx=(0, 7))
        self._button(right, "Jump to end", self.jump_to_end).pack(
            side="left", padx=(0, 16)
        )

        tk.Label(
            right,
            text="Speed",
            bg=SURFACE,
            fg=MUTED,
            font=self.fonts["small"],
        ).pack(side="left", padx=(0, 4))
        self.speed_scale = tk.Scale(
            right,
            from_=1,
            to=30,
            orient="horizontal",
            variable=self.speed_var,
            showvalue=False,
            bg=SURFACE,
            fg=SECONDARY,
            troughcolor="#4a4a46",
            activebackground=PRIMARY,
            highlightthickness=0,
            bd=0,
            length=142,
            sliderlength=14,
            width=7,
            command=self._on_speed,
        )
        self.speed_scale.pack(side="left")
        self.speed_label = tk.Label(
            right,
            text=f"{self.speed_var.get()} steps/s",
            bg=SURFACE,
            fg=SECONDARY,
            font=self.fonts["small"],
            width=8,
            anchor="w",
        )
        self.speed_label.pack(side="left", padx=(4, 0))

    def _spinbox(
        self,
        parent: tk.Misc,
        variable: tk.StringVar,
        *,
        from_: int,
        to: int,
        width: int,
    ) -> tk.Spinbox:
        return tk.Spinbox(
            parent,
            from_=from_,
            to=to,
            textvariable=variable,
            width=width,
            bg=SURFACE_DEEP,
            fg=PRIMARY,
            insertbackground=PRIMARY,
            buttonbackground=SURFACE_RAISED,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=SECONDARY,
            font=self.fonts["input"],
            justify="center",
        )

    def _button(
        self,
        parent: tk.Misc,
        text: str,
        command: Any,
        *,
        primary: bool = False,
    ) -> tk.Button:
        bg = PRIMARY if primary else SURFACE_RAISED
        fg = PAGE if primary else PRIMARY
        active_bg = "#ffffff" if primary else BORDER
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=13,
            pady=6,
            cursor="hand2",
            font=self.fonts["button"],
        )
        button.bind(
            "<Enter>",
            lambda _event: button.configure(bg=active_bg),
        )
        button.bind(
            "<Leave>",
            lambda _event: button.configure(bg=bg),
        )
        return button

    def _build_panels(self) -> None:
        main = tk.Frame(self.root, bg=PAGE)
        main.grid(row=2, column=0, sticky="nsew", padx=18)
        for row in range(2):
            main.grid_rowconfigure(row, weight=1, uniform="row")
        for column in range(2):
            main.grid_columnconfigure(column, weight=1, uniform="panel")

        self.panels: dict[str, MazePanel] = {}
        for index, spec in enumerate(ALL_SPECS):
            row, column = divmod(index, 2)
            panel = MazePanel(
                main,
                spec,
                fonts=self.fonts,
                show_gradient=spec.key == "shortest",
            )
            padx = (6, 3) if column == 0 else (3, 6)
            pady = (3, 3)
            panel.card.grid(
                row=row,
                column=column,
                sticky="nsew",
                padx=padx,
                pady=pady,
            )
            self.panels[spec.key] = panel
        self.shortest_panel = self.panels["shortest"]

    def _build_footer(self) -> None:
        footer = tk.Frame(self.root, bg=PAGE)
        footer.grid(row=3, column=0, sticky="ew", padx=28, pady=(8, 14))
        footer.grid_columnconfigure(0, weight=1)

        legend = tk.Frame(footer, bg=PAGE)
        legend.grid(row=0, column=0, sticky="w")
        tk.Label(
            legend,
            text="Mode",
            bg=PAGE,
            fg=MUTED,
            font=self.fonts["footer"],
        ).pack(side="left", padx=(0, 8))
        for spec in ALL_SPECS:
            tk.Label(
                legend,
                text="━",
                bg=PAGE,
                fg=spec.color,
                font=self.fonts["footer"],
            ).pack(side="left", padx=(0, 3))
            tk.Label(
                legend,
                text=(
                    "shortest (oracle)"
                    if spec.key == "shortest"
                    else spec.key
                ),
                bg=PAGE,
                fg=SECONDARY,
                font=self.fonts["footer"],
            ).pack(side="left", padx=(0, 10))
        tk.Label(
            legend,
            text=(
                "Fair: dark=unknown · tinted=explored · warm=known wall · "
                "×=bump · white dot=car · S=start · pink diamond=goal  |  "
                "Gradient: blue=closer · number=steps · arrow=downhill"
            ),
            bg=PAGE,
            fg=MUTED,
            font=self.fonts["footer"],
        ).pack(side="left", padx=(6, 0))

        self.status_label = tk.Label(
            footer,
            textvariable=self.status_var,
            bg=PAGE,
            fg=SECONDARY,
            font=self.fonts["footer"],
            anchor="w",
        )
        self.status_label.grid(row=1, column=0, sticky="w", pady=(7, 0))

    # ------------------------------------------------------------- lifecycle
    def _read_controls(self) -> tuple[int, int] | None:
        try:
            seed = int(self.seed_var.get().strip())
            size = int(self.size_var.get().strip())
        except ValueError:
            messagebox.showwarning(
                "Invalid input", "Seed and size must be integers.", parent=self.root
            )
            return None
        if seed < 0:
            messagebox.showwarning(
                "Invalid input", "Seed must be a non-negative integer.", parent=self.root
            )
            return None
        if not 4 <= size <= 20:
            messagebox.showwarning(
                "Invalid input", "Demo size must be between 4 and 20.", parent=self.root
            )
            return None
        return seed, size

    def _regenerate(self) -> None:
        controls = self._read_controls()
        if controls is None:
            return
        seed, size = controls
        self.load_run(seed=seed, size=size, autoplay=True)

    def _randomize(self) -> None:
        seed = random.randrange(0, 10_000)
        if seed == self.seed:
            seed = (seed + 1) % 10_000
        self.seed_var.set(str(seed))
        controls = self._read_controls()
        if controls is None:
            return
        _, size = controls
        self.load_run(seed=seed, size=size, autoplay=True)

    def _cancel_tick(self) -> None:
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def load_run(self, *, seed: int, size: int, autoplay: bool) -> None:
        self._cancel_tick()
        self.playing = False
        self.status_var.set("Generating traces...")
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            traces = build_traces(size=size, seed=seed)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.root.configure(cursor="")
            self.status_var.set("Generation failed")
            messagebox.showerror(
                "Generation failed", f"Could not build the demo: {exc}", parent=self.root
            )
            return

        self.root.configure(cursor="")
        self.seed = seed
        self.size = size
        self.seed_var.set(str(seed))
        self.size_var.set(str(size))
        self.traces = traces
        self.index = 0
        self.max_index = max(
            len(trace.frames) - 1 for trace in traces.values()
        )
        for key, panel in self.panels.items():
            panel.set_trace(traces[key])

        self.playing = autoplay
        self._update_controls()
        self._update_status()
        if self.playing:
            self._schedule_tick()

    def _schedule_tick(self) -> None:
        self._cancel_tick()
        if not self.playing or self.index >= self.max_index:
            return
        delay = max(16, round(1000 / max(1, self.speed_var.get())))
        self.after_id = self.root.after(delay, self._on_tick)

    def _on_tick(self) -> None:
        self.after_id = None
        if not self.playing:
            return
        if self.index >= self.max_index:
            self._finish()
            return
        self.index += 1
        self._refresh_panels()
        if self.index >= self.max_index:
            self._finish()
        else:
            self._update_status()
            self._schedule_tick()

    def _refresh_panels(self) -> None:
        for panel in self.panels.values():
            panel.set_index(self.index)

    def _finish(self) -> None:
        self._cancel_tick()
        self.playing = False
        self.index = self.max_index
        self._refresh_panels()
        self._update_controls()
        self._update_status()

    def _update_controls(self) -> None:
        if self.playing:
            label = "Pause"
        elif self.index >= self.max_index:
            label = "Replay"
        else:
            label = "Continue"
        self.play_button.configure(text=label)

    def _summary(self) -> str:
        if self.traces is None:
            return ""
        parts = []
        for spec in ALL_SPECS:
            trace = self.traces[spec.key]
            marker = "" if trace.reached else " (not reached)"
            parts.append(f"{spec.key} {trace.final_steps} steps{marker}")
        return "Demo complete · " + " · ".join(parts)

    def _update_status(self) -> None:
        if self.traces is None:
            return
        head = (
            f"4 modes · seed {self.seed} · "
            f"{self.size}×{self.size}"
        )
        if self.index >= self.max_index and not self.playing:
            self.status_var.set(f"{head} · {self._summary()}")
        elif self.playing:
            self.status_var.set(
                f"{head} · playing · {self.index}/{self.max_index}"
            )
        else:
            self.status_var.set(
                f"{head} · paused · {self.index}/{self.max_index}"
            )

    # --------------------------------------------------------------- actions
    def toggle_play(self) -> None:
        if self.traces is None:
            return
        if self.playing:
            self.playing = False
            self._cancel_tick()
        else:
            if self.index >= self.max_index:
                self.index = 0
                self._refresh_panels()
            self.playing = True
            self._schedule_tick()
        self._update_controls()
        self._update_status()

    def step_once(self) -> None:
        if self.traces is None:
            return
        self.playing = False
        self._cancel_tick()
        if self.index < self.max_index:
            self.index += 1
            self._refresh_panels()
        self._update_controls()
        self._update_status()

    def jump_to_end(self) -> None:
        if self.traces is None:
            return
        self.playing = False
        self._cancel_tick()
        self.index = self.max_index
        self._refresh_panels()
        self._update_controls()
        self._update_status()

    def _on_speed(self, _value: str) -> None:
        try:
            self.speed_label.configure(text=f"{self.speed_var.get()} steps/s")
        except tk.TclError:
            pass

    def close(self) -> None:
        self._cancel_tick()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point


def _enable_high_dpi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001 - optional cosmetic improvement
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graphical (Tkinter) 2x2 maze exploration demo"
    )
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--size", type=int, default=14)
    parser.add_argument("--speed", type=int, default=14, help="steps per second")
    parser.add_argument(
        "--paused", action="store_true", help="open paused on the first frame"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 4 <= args.size <= 20:
        print("gui_demo: --size must be between 4 and 20", file=sys.stderr)
        return 2
    if args.seed < 0:
        print("gui_demo: --seed must be non-negative", file=sys.stderr)
        return 2
    speed = max(1, min(30, args.speed))

    _enable_high_dpi()
    root = tk.Tk()
    DemoApp(
        root,
        seed=args.seed,
        size=args.size,
        speed=speed,
        autoplay=not args.paused,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
