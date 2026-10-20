from edtrace import link, make_graph
from typing import Callable, Any
import altair as alt
from altair import Chart, Data
import base64
import functools
import hashlib
import inspect
import json
import os
import pickle
import types
import numpy as np
import random
import torch


def article_link(url):
    return link(url, title="[article]")


PLOT_SIZE = 300  # Width and height of plots with an arrow, in pixels


def make_plot(title: str | None,
              xlabel: str,
              ylabel: str,
              f: Callable[[float], float] | None,
              xrange: tuple[float, float] = (-3, 3),
              points: list[dict] | None = None,
              line_color: str | None = None,
              above_color: str | None = None,
              below_color: str | None = None,
              arrow: tuple[tuple[float, float], tuple[float, float]] | None = None,
              num_points: int = 30,
              domain: tuple[float, float] | None = None) -> dict:
    """
    If `above_color` or `below_color` is given (requires `points` or `domain`), shade the region above or below the line `f`.
    If `arrow` = (start, end) is given, draw an arrow from start to end (e.g., the weight vector).
    Plots with an arrow use the same scale on both axes, so that angles are drawn faithfully
    (e.g., the weight vector is perpendicular to the decision boundary).
    `num_points` is the number of points at which `f` is sampled (increase for sharp transitions).
    If `domain` is given, use it as the range of both axes.
    """
    to_show = []

    values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(xrange[0], xrange[1], num_points)] if f is not None else []

    # If there are points, pad both axes so that points near the edges are easy to see
    x = alt.X(f"{xlabel}:Q")
    y = alt.Y(f"{ylabel}:Q")
    if domain is not None:
        x_domain, y_domain = list(domain), list(domain)
        x = x.scale(domain=x_domain)
        y = y.scale(domain=y_domain)
        if f is not None:  # Extend the line across the full x-axis (clipped to the y-axis below)
            values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(x_domain[0], x_domain[1], num_points)]
    elif points:
        if arrow is not None:
            # Fit the points and the arrow (the line is clipped), with equal scales on both axes
            domain_points = points + [{xlabel: ax, ylabel: ay} for ax, ay in arrow]
            x_domain = padded_domain([p[xlabel] for p in domain_points])
            y_domain = padded_domain([p[ylabel] for p in domain_points])
            x_domain, y_domain = equalize_spans(x_domain, y_domain)
        else:
            x_domain = padded_domain([p[xlabel] for p in points + values])
            y_domain = padded_domain([p[ylabel] for p in points + values])
        x = x.scale(domain=x_domain)
        y = y.scale(domain=y_domain)
        if f is not None:  # Extend the line across the full x-axis (clipped to the y-axis below)
            values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(x_domain[0], x_domain[1], num_points)]

    if f is not None and (points or domain is not None):
        # Shade regions above/below the line (fill up to the edge of the plot)
        for color, edge in [(above_color, y_domain[1]), (below_color, y_domain[0])]:
            if color is not None:
                region = Chart(Data(values=values)).mark_area(clip=True, color=color, opacity=0.3).encode(x=x, y=y, y2=alt.Y2(datum=edge))
                to_show.append(region)

    if f is not None:
        line = Chart(Data(values=values)).mark_line(clip=True, **({"color": line_color} if line_color else {})).encode(x=x, y=y)
        to_show.append(line)

    if points is not None:
        points = Chart(Data(values=points)).mark_point(filled=True, size=100, opacity=1).encode(x=x, y=y, color=alt.Color("color:N").scale(None))
        to_show.append(points)

    if arrow is not None:
        to_show.extend(make_arrow(arrow, xlabel, ylabel, x, y, color=line_color or "black"))

    chart = functools.reduce(lambda c1, c2: c1 + c2, to_show)
    if arrow is not None or domain is not None:
        chart = chart.properties(width=PLOT_SIZE, height=PLOT_SIZE)
    if title is not None:
        chart = chart.properties(title=title)
    return chart.to_dict()


def make_arrow(arrow: tuple[tuple[float, float], tuple[float, float]], xlabel: str, ylabel: str,
               x: alt.X, y: alt.Y, color: str) -> list[Chart]:
    """Return the charts (shaft and head) for an arrow from `arrow[0]` to `arrow[1]`."""
    (x0, y0), (x1, y1) = arrow
    # Pixels per unit (plot is PLOT_SIZE x PLOT_SIZE pixels), used to orient the head on screen
    x_domain = x.to_dict().get("scale", {}).get("domain")
    y_domain = y.to_dict().get("scale", {}).get("domain")
    x_pixels_per_unit = PLOT_SIZE / (x_domain[1] - x_domain[0]) if x_domain else 1
    y_pixels_per_unit = PLOT_SIZE / (y_domain[1] - y_domain[0]) if y_domain else 1
    dx_pixels, dy_pixels = (x1 - x0) * x_pixels_per_unit, (y1 - y0) * y_pixels_per_unit
    angle = np.degrees(np.arctan2(dx_pixels, dy_pixels)) % 360  # Clockwise from up, in [0, 360)

    # Custom triangle whose tip is at the anchor (0, 0), so the tip lands exactly on the end point
    head_size = 150
    head_shape = "M0,0 L-0.6,1.5 L0.6,1.5 Z"
    head_length_pixels = 1.5 * np.sqrt(head_size) / 2  # Path coordinates are scaled by sqrt(size) / 2
    head = Chart(Data(values=[{xlabel: x1, ylabel: y1}])).mark_point(shape=head_shape, filled=True, size=head_size, opacity=1, color=color, strokeWidth=0, angle=float(angle)).encode(x=x, y=y)

    # Stop the shaft inside the head so its thick end doesn't poke out past the tip
    shrink = max(0, 1 - 0.6 * head_length_pixels / np.hypot(dx_pixels, dy_pixels))
    shaft_end = {xlabel: x0 + (x1 - x0) * shrink, ylabel: y0 + (y1 - y0) * shrink}
    shaft = Chart(Data(values=[{xlabel: x0, ylabel: y0}, shaft_end])).mark_line(color=color, strokeWidth=3).encode(x=x, y=y)
    return [shaft, head]


def equalize_spans(x_domain: list[float], y_domain: list[float]) -> tuple[list[float], list[float]]:
    """Widen the narrower of the two domains (around its center) so both have the same span."""
    span = max(x_domain[1] - x_domain[0], y_domain[1] - y_domain[0])
    def widen(domain: list[float]) -> list[float]:
        center = (domain[0] + domain[1]) / 2
        return [center - span / 2, center + span / 2]
    return widen(x_domain), widen(y_domain)


def padded_domain(values: list[float], fraction: float = 0.15) -> list[float]:
    """Return [min, max] of `values`, extended on both sides by `fraction` of the range (at least 1)."""
    low, high = float(min(values)), float(max(values))
    pad = max((high - low) * fraction, 1)
    return [low - pad, high + pad]


class Vocabulary:
    """Maps strings to integers."""
    def __init__(self):
        self.index_to_string: list[str] = []
        self.string_to_index: dict[str, int] = {}

    def get_index(self, string: str) -> int:  # @inspect string
        index = self.string_to_index.get(string)  # @inspect index
        if index is None:  # New string
            index = len(self.index_to_string)  # @inspect index
            self.index_to_string.append(string)
            self.string_to_index[string] = index
        return index

    def get_string(self, index: int) -> str:
        return self.index_to_string[index]

    def __len__(self):
        return len(self.index_to_string)

    def asdict(self):
        return {
            "index_to_string": self.index_to_string,
            "string_to_index": self.string_to_index,
        }


def cached(func: Callable) -> Callable:
    """
    Wrap `func` so that calls are cached in var/func_cache/<caller>-<hash>.json (<caller> is the calling file, e.g., function_approximation), e.g., `cached(generate_rollouts_rl)(mdp, agent, num_trials=50)`.
    The hash covers the source code of `func` and of the arguments' classes, the full state of the arguments, and the random number generator states.
    Since `func` might mutate its arguments (e.g., the agent learns) and consumes randomness,
    the cache also stores the arguments' state and the random number generator states after the call, and restores them on a cache hit.
    Caveat: changes to other code that `func` calls aren't detected (delete var/func_cache to reset).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        objects = [x for x in list(args) + list(kwargs.values()) if has_state(x)]  # Arguments that might be mutated
        key = {
            "function": f"{func.__module__}.{func.__qualname__}",
            "source": [inspect.getsource(func)] + [get_class_source(type(x)) for x in objects],
            "args": fingerprint(args),
            "kwargs": fingerprint(kwargs),
            "rng": fingerprint(get_rng_states()),
        }
        hash = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        caller = os.path.splitext(os.path.basename(inspect.currentframe().f_back.f_code.co_filename))[0]  # e.g., function_approximation
        path = os.path.join("var", "func_cache", f"{caller}-{hash}.json")

        if os.path.exists(path):
            # Cache hit: restore the result, the arguments' state, and the random number generator states
            with open(path) as f:
                state = pickle.loads(base64.b64decode(json.load(f)["state"]))
            for x, x_dict in zip(objects, state["objects"]):
                x.__dict__.clear()
                x.__dict__.update(x_dict)
            set_rng_states(state["rng"])
            return state["result"]

        # Cache miss: call the function and save everything
        result = func(*args, **kwargs)
        state = {"result": result, "objects": [x.__dict__ for x in objects], "rng": get_rng_states()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "function": key["function"],
                "result": result if is_json_serializable(result) else repr(result),  # For human inspection
                "state": base64.b64encode(pickle.dumps(state)).decode(),
            }, f)
        return result
    return wrapper


def has_state(x: Any) -> bool:
    """Whether `x` is an object with mutable state (attributes)."""
    return hasattr(x, "__dict__") and not isinstance(x, (type, types.ModuleType, types.FunctionType, types.MethodType, functools.partial))


def get_class_source(cls: type) -> str:
    """Return the source code of `cls` and its base classes (if available)."""
    sources = []
    for c in cls.__mro__:
        try:
            sources.append(inspect.getsource(c))
        except (TypeError, OSError):
            pass  # Built-in classes have no source
    return "\n".join(sources)


def get_rng_states() -> dict:
    return {"random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_states(states: dict):
    random.setstate(states["random"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])


def is_json_serializable(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


def fingerprint(x: Any, seen: set | None = None) -> Any:
    """Return a deterministic JSON-serializable representation of `x` (for hashing)."""
    seen = set() if seen is None else seen
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return repr(x)
    if id(x) in seen:  # Avoid infinite recursion on cycles
        return ["cycle", type(x).__qualname__]
    seen = seen | {id(x)}
    if isinstance(x, torch.Tensor):
        return ["tensor", str(x.dtype), x.tolist()]
    if isinstance(x, np.ndarray):
        return ["ndarray", str(x.dtype), x.tolist()]
    if isinstance(x, np.generic):
        return fingerprint(x.item(), seen)
    if isinstance(x, (list, tuple)):
        return [type(x).__qualname__, [fingerprint(y, seen) for y in x]]
    if isinstance(x, (set, frozenset)):
        return [type(x).__qualname__, sorted(json.dumps(fingerprint(y, seen)) for y in x)]
    if isinstance(x, dict):  # Keep the order (it can matter, e.g., for breaking ties)
        return [type(x).__qualname__, [[fingerprint(k, seen), fingerprint(v, seen)] for k, v in x.items()]]
    if isinstance(x, functools.partial):
        return ["partial", fingerprint(x.func, seen), fingerprint(x.args, seen), fingerprint(x.keywords, seen)]
    if isinstance(x, types.MethodType):
        return ["method", x.__func__.__qualname__, fingerprint(x.__self__, seen)]
    if isinstance(x, (types.FunctionType, types.BuiltinFunctionType, type)):
        return ["function", x.__module__, x.__qualname__]
    if isinstance(x, torch.nn.Module):
        return ["module", type(x).__qualname__, fingerprint(dict(x.state_dict()), seen)]
    if isinstance(x, torch.optim.Optimizer):
        return ["optimizer", type(x).__qualname__, fingerprint(x.state_dict(), seen)]
    if hasattr(x, "__dict__"):
        return ["object", type(x).__module__, type(x).__qualname__, fingerprint(vars(x), seen)]
    return ["repr", repr(x)]


def set_random_seed(seed: int):
    """Set all random seeds for deterministic behavior."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def one_hot(index: int, length: int) -> torch.Tensor:
    """Create a one-hot vector of the given `length` with a 1 at the `index` position."""
    vector = torch.zeros(length)
    vector[index] = 1
    return vector


def sample_dict(choices: dict[Any, float]) -> Any:
    """Sample a key from a dictionary of choices based on their probabilities (values)."""
    return np.random.choice(list(choices.keys()), p=list(choices.values()))


def normalize_dict(choices: dict[Any, float]) -> dict[Any, float]:
    """Normalize a dictionary of choices based on their probabilities (values)."""
    total_prob = sum(choices.values())
    return {key: prob / total_prob for key, prob in choices.items()}


# Graphs of search problems (drawn with `graph`)

def search_graph_stylesheet() -> list[dict]:
    """Stylesheet for graphs of search problems."""
    return [
        {"selector": "node.end", "style": {"border-style": "double", "border-width": 6}},  # End states: double circle
        {"selector": "edge.walk", "style": {"line-color": "#2a9d8f", "target-arrow-color": "#2a9d8f", "color": "#2a9d8f"}},
        {"selector": "edge.tram", "style": {"line-color": "#9b72cf", "target-arrow-color": "#9b72cf", "color": "#9b72cf"}},
        {"selector": "edge[curve]", "style": {"curve-style": "unbundled-bezier", "control-point-weights": 0.5,
                                              "control-point-distances": "data(curve)"}},
        # Solution path: thicker, bright orange, drawn on top
        {"selector": "edge.path", "style": {"width": 4, "line-color": "#f77f00", "target-arrow-color": "#f77f00", "color": "#f77f00",
                                            "z-index": 10}},
        # Numbers (e.g., costs) next to nodes: separate, unclickable label nodes
        {"selector": "node.annotation", "style": {"width": 1, "height": 1, "background-opacity": 0, "border-width": 0,
                                                  "color": "#f77f00", "font-weight": "bold", "events": "no"}},
    ]


def draw_rollouts(problem, solutions: list, solution_only: bool = False, draw_graph: Callable | None = None) -> dict | None:
    """
    Return a graph showing `solutions` (lists of steps from the start state of the search `problem`) to show with `graph`:
    - If the states are strings (e.g., prompt + generated text for language models): show the tree of solutions,
      with what each step adds to the text in the nodes and costs on the edges.
    - If `solution_only`: show just the states and steps of the solutions (as a tree, sharing common prefixes),
      with the last state of each labeled with its cost.
    - Otherwise: a copy of the full graph for each solution (from `draw_graph(problem, solution, state_costs)`), stacked vertically,
      with the solution's path in orange and each state on it labeled with the cost remaining along the solution
      (None if `draw_graph` isn't given, which `graph` shows as nothing).
    """
    if isinstance(problem.start_state(), str):
        return draw_solution_tree(problem, solutions, edge_label=lambda step: f"{step.cost:.2f}", text_nodes=True)
    if solution_only:
        return draw_solution_tree(problem, solutions)
    if draw_graph is None:
        return None
    specs = []
    for solution in solutions:
        # Cost remaining from each state on the path (the sum of the costs of the rest of the steps)
        states = [problem.start_state()] + [step.state for step in solution.steps]
        costs = [step.cost for step in solution.steps]
        remaining_costs = {state: sum(costs[i:]) for i, state in enumerate(states)}
        specs.append(draw_graph(problem, solution, state_costs=remaining_costs))
    return stack_graphs(specs, spacing=150)


def draw_solution_tree(problem, solutions: list,
                       edge_label: Callable = lambda step: f"{step.action[0].upper()}:{step.cost}",
                       text_nodes: bool = False) -> dict:
    """
    Return the graph of just the states and steps of `solutions` (steps in orange), with the last state of each solution labeled with its cost.
    Solutions that share a prefix (the same step objects, e.g., candidates extended from the same candidate) share its nodes,
    so this is a tree rooted at the start state, growing to the right.
    - `edge_label(step)`: how to label the edge for `step`
    - `text_nodes`: states are strings that extend each other (e.g., prompt + generated text): show the start state in full,
      and each other state as just what it adds to the previous one, in rounded rectangles
    """
    # Build the tree: a node per distinct prefix of steps (identified by the step objects)
    root = {"id": 0, "state": problem.start_state(), "depth": 0, "children": [], "step": None, "parent": None}
    prefix_to_node = {(): root}
    end_costs = {}  # Node id -> cost of a solution that ends there
    for solution in solutions:
        prefix, node = (), root
        for step in solution.steps:
            prefix = prefix + (id(step),)
            if prefix not in prefix_to_node:
                child = {"id": len(prefix_to_node), "state": step.state, "depth": node["depth"] + 1, "children": [], "step": step, "parent": node}
                prefix_to_node[prefix] = child
                node["children"].append(child)
            node = prefix_to_node[prefix]
        end_costs[node["id"]] = solution.cost

    def node_label(node) -> str:
        if not text_nodes or node["parent"] is None:
            return str(node["state"])
        # Just what this state adds to the previous one (making whitespace visible)
        added = node["state"][len(node["parent"]["state"]):]
        return added.replace("\n", "⏎").strip() or "␣"

    # Lay out the tree from left to right: leaves stacked vertically, each parent centered beside its children
    leaf_spacing, depth_spacing = (44, 120) if text_nodes else (60, 80)
    num_leaves = 0
    nodes, edges = [], []
    def layout(node) -> float:
        nonlocal num_leaves
        if node["children"]:
            y = sum(layout(child) for child in node["children"]) / len(node["children"])
        else:
            y = leaf_spacing * num_leaves
            num_leaves += 1
        x = depth_spacing * node["depth"] - (30 if text_nodes and node["parent"] is None else 0)  # Room for the (wider) prompt
        classes = ("end " if problem.is_end(node["state"]) else "") + ("text" if text_nodes else "") + (" prompt" if text_nodes and node["parent"] is None else "")
        nodes.append({"id": node["id"], "label": node_label(node), "x": x, "y": y, "classes": classes.strip()})
        if node["id"] in end_costs:
            cost = end_costs[node["id"]]
            cost_label = f"{cost:.2f}" if isinstance(cost, float) else str(cost)
            cost_position = (x + 68, y) if text_nodes else (x + 20, y - 24)  # Right of text nodes, upper right of circles
            nodes.append({"id": f"{node['id']}-cost", "label": cost_label, "x": cost_position[0], "y": cost_position[1], "classes": "annotation"})
        for child in node["children"]:
            step = child["step"]
            classes = "path" if text_nodes else f"{step.action} path"  # Color by action (walk/tram) if it's a name
            edges.append({"source": node["id"], "target": child["id"], "label": edge_label(step), "classes": classes})
        return y
    layout(root)

    max_depth = max(node["depth"] for node in prefix_to_node.values())
    stylesheet = search_graph_stylesheet()
    if text_nodes:
        # Rounded rectangles with the text (the prompt is wider)
        stylesheet += [
            {"selector": "node.text", "style": {"shape": "round-rectangle", "width": 76, "height": 28, "font-size": 12,
                                                "text-wrap": "ellipsis", "text-max-width": "72px"}},
            {"selector": "node.prompt", "style": {"width": 116, "text-max-width": "112px"}},
        ]
        # Shrink wide trees to fit (keeping the proportions)
        width, height = depth_spacing * max_depth + 200, leaf_spacing * num_leaves + 40
        scale = min(1, 760 / width)
        return make_graph(nodes, edges, stylesheet=stylesheet, width=round(width * scale), height=round(height * scale))
    return make_graph(nodes, edges, stylesheet=stylesheet, width=min(760, 80 * max_depth + 120), height=60 * num_leaves + 60)


def stack_graphs(specs: list[dict], spacing: float) -> dict:
    """Combine graphs (from `make_graph`) into one, stacking them vertically `spacing` apart (they share the first stylesheet)."""
    nodes, edges = [], []
    for i, spec in enumerate(specs):
        prefix = f"g{i}-"  # Keep the ids of the copies distinct
        for node in spec["nodes"]:
            node = {**node, "data": {**node["data"], "id": prefix + node["data"]["id"]}}
            if "position" in node:
                node["position"] = {"x": node["position"]["x"], "y": node["position"]["y"] + spacing * i}
            nodes.append(node)
        for edge in spec["edges"]:
            edges.append({**edge, "data": {**edge["data"], "source": prefix + edge["data"]["source"], "target": prefix + edge["data"]["target"]}})
    return {**specs[0], "nodes": nodes, "edges": edges, "height": specs[0]["height"] * len(specs)}
