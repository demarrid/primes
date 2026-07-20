from vispy.color import ColorArray
from monzo import Monzo, color_sequence
import networkx as nx
from visualization import draw_collatz_filter, draw_collatz_graph
import numpy as np
from vispy import app
import time
from clanker_utils import expand_doublings
from utils import pos_mod_coords_to_string
import pandas as pd

G = nx.DiGraph()

collatz_nodes = []
start_nodes = []
collatz_edges = []
predecessor_nodes = []
source_nodes = []

odd_teal = ColorArray("#6ca6d9").rgba
evil_sable = ColorArray("#666666").rgba
source = ColorArray("#edb193").rgba
pure_source = ColorArray("#FFFFFF").rgba
green_goblin = ColorArray("#1ced1c").rgba
orange_obviant = ColorArray("#DB782A").rgba

def improvement(n, nxt):
    n = 3 * n + 1.0
    nxt = 3 * nxt + 1.0
    n = np.log2(n) % 1
    nxt = np.log2(nxt) % 1
    return abs(nxt - n)

show_odd_nodes = True


def do_collatz(n):
    if n <= 0:
        return
    if show_odd_nodes:
        do_odd_collatz(n)
    else:
        do_even_collatz(n)

def do_even_collatz(n):

    visited = set()
    while n != 1 and n != 4 and n > 0:
        if n in visited:
            break
        visited.add(n)

        if n not in collatz_nodes:
            collatz_nodes.append(n)

        k = n
        while k % 2 == 0:
            k //= 2

        nxt = 3 * k + 1

        collatz_edges.append((n, nxt))

        if nxt in collatz_nodes:
            break

        n = nxt

    if n > 0 and n not in collatz_nodes:
        collatz_nodes.append(n)

def do_odd_collatz(n):
    while n > 0 and n % 2 == 0:
        n //= 2

    while n != 1:
        nxt = 3 * n + 1
        while nxt % 2 == 0:
            nxt //= 2

        collatz_edges.append((n, nxt))
        if nxt in collatz_nodes:
            break
        collatz_nodes.append(nxt)
        n = nxt


def predecessors_of(m: int, count: int):
    to_return = []

    if show_odd_nodes:
        while m > 0 and m % 2 == 0:
            m //= 2

        k = m % 3

        if k == 0:
            source_nodes.append(m)
            return to_return

        for j in range(1, count + 1):
            new_2_index = 2 * (j + 1) + k - 1
            value_with_new_2_index = m * (2**new_2_index)
            z = (value_with_new_2_index - 1) // 3

            to_return.append(int(z))

        return to_return
    else:
        if m % 2 == 1:
            m *= 3
            m += 1

        k = m % 3

        if k != 1:
            return to_return

        t = (m - 1) // 3

        for j in range(1, count + 1):
            z = t * 2**j
            to_return.append(z)

        return to_return

def get_monzo_desc(monzo: Monzo):
    mod_coords_str = (
            "[" + ", ".join(str(j) for j in monzo.get_extended_modular_coordinates()) + "]"
    )
    if len(mod_coords_str) > 50:
        mod_coords_str = mod_coords_str[:25] + "..." + mod_coords_str[-25:]

    coords_str = "‹" + ", ".join([str(c) for c in monzo.get_coordinates()]) + "›"

    if len(coords_str) > 50:
        coords_str = coords_str[:25] + "..." + coords_str[-25:]

    return f"‖m‖²: {monzo.square_norm()}\n|m_P|: {len(monzo)}\nm_P(+): {pos_mod_coords_to_string(monzo)}\nm_P(-): {mod_coords_str}\nm:{coords_str}\nm_Z: {monzo.to_int()}"

def reverse_collatz(sink: int, width: int, depth: int):
    if show_odd_nodes and sink % 3 == 0:
        source_nodes.append(sink)
        do_collatz(sink)
    elif not show_odd_nodes and sink % 3 != 1:
        source_nodes.append(sink)
        do_collatz(sink)
    elif depth > 1:
        for p in predecessors_of(sink, width):
            reverse_collatz(p, width, depth - 1)
    else:
        start_nodes.append(sink)
        preds = predecessors_of(sink, width)
        for p in preds:
            predecessor_nodes.append(p)
            do_collatz(p)

def graph():
    t = time.time()

    # reverse_collatz(1, 8, 4)
    do_collatz(1875)
    do_collatz(1874)
    do_collatz(1895)

    print( f"Collatz completed, size of graph: {len(collatz_nodes) + len(predecessor_nodes)} nodes, {len(collatz_edges)} edges")

    print(f"Time taken: {time.time() - t}s")
    t = time.time() 

    G.add_edges_from(collatz_edges)
    G.graph["rankdir"] = "BT"
    G.graph["nodesep"] = 1.0
    G.graph["overlap_scaling"] = 2
    G.graph["ranksep"] = "2.0"

    # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    pos = nx.nx_agraph.graphviz_layout(G.reverse(), prog="twopi", root=1)
    # pos = nx.nx_agraph.graphviz_layout(G, prog="neato")

    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    xy = np.array([pos[n] for n in nodes], dtype=float)

    face = np.full((len(nodes), 4), 0.5)



    print(f"Graph organization took ({time.time() - t}s)")

    t = time.time()
    for i, n in enumerate(nodes):
        if n in source_nodes:
            if n in start_nodes:
                face[i] = source
            else:
                face[i] = pure_source
        elif n in start_nodes:
            face[i] = evil_sable
        elif n in predecessor_nodes:
            face[i] = green_goblin
        elif n in collatz_nodes:
            face[i] = odd_teal if n % 3 == 1 else orange_obviant

    def label_fn(i, data):
        if nodes[i] > Monzo.get_prime_of_index(-1):
            return str(nodes[i])

        monzo = Monzo.from_int(nodes[i])
        return get_monzo_desc(monzo)

    print(f"Pre-graph took ({time.time() - t}s)")

    draw_collatz_graph(collatz_edges, xy, face, index, label_fn=label_fn)
    app.run()

p2_cache = {}

above_cache = {}

def project_above_p2(n):
    n = int(n)
    if n in above_cache:
        return above_cache[n]
    above_cache[n] = n >> get_p2_index(n)
    return above_cache[n]

def get_p2_index(n):
    n = int(n)
    if n in p2_cache:
        return p2_cache[n]
    original = n
    if n <= 1:
        p2_cache[original] = 0
        return 0
    i = 0
    while n % 2 == 0:
        n //= 2
        i += 1
    p2_cache[original] = i
    return i

def filter():
    CACHE_RANGE_OFFSET = 12

    remaining = set()
    reported_clears = set()
    cached_bounds = None
    removed_levels = []
    color_levels = []
    r_to_color = {}

    low_opacity_black = ColorArray(
        (0, 0, 0),
        alpha=0.1,
    ).rgba

    def reset_cache(start, stop):
        nonlocal cached_bounds
        nonlocal removed_levels
        nonlocal color_levels
        nonlocal r_to_color
        nonlocal remaining, reported_clears

        remaining = set(range(1, stop + 1))
        reported_clears = set()

        cached_bounds = (start, stop)
        removed_levels = []
        color_levels = []
        r_to_color = {}

    def color_for(r):
        key = project_above_p2(int(r))

        if key not in r_to_color:
            i = len(r_to_color)
            r_to_color[key] = ColorArray(
                color_sequence[i % len(color_sequence)]
            ).rgba

        return r_to_color[key]

    def calculate_level(depth, stop):
        nonlocal r_to_color

        removed = set()
        colors = {}

        if depth == 0:
            for exponent in range(
                1,
                stop.bit_length() + CACHE_RANGE_OFFSET,
            ):
                value = 1 << exponent
                removed.add(value)
                colors[value] = source

            return removed, colors

        previous = removed_levels[depth - 1]

        if depth == 1:
            bases = []

            for value in previous:
                numerator = value - 1

                if numerator % 3 == 0:
                    base = numerator // 3
                    if base > 0:
                        bases.append(base)

            values, _ = expand_doublings(
                bases,
                stop,
                range_offset=CACHE_RANGE_OFFSET,
            )

            for value in values:
                value = int(value)

                if value in previous:
                    continue

                removed.add(value)
                colors.setdefault(value, orange_obviant)

            odd_values = sorted(
                value
                for value in removed
                if value > 1 and value % 2 == 1
            )

            r_to_color = {
                value: ColorArray(
                    color_sequence[i % len(color_sequence)]
                ).rgba
                for i, value in enumerate(odd_values)
            }

            return removed, colors

        if depth == 2:
            bases = []
            source_values = []

            for r in previous:
                if r % 3 != 2:
                    continue

                max_p = int(
                    np.log2((3 * stop + 1) / r)
                )

                for p in range(max_p + 1):
                    if p > 1 and (
                        p % r == 0 or r % p == 0
                    ):
                        continue

                    numerator = (1 << p) * r - 1

                    if numerator % 3 != 0:
                        continue

                    base = numerator // 3

                    if base > 0 and base % 2 == 1:
                        bases.append(base)
                        source_values.append(r)

            values, source_indices = expand_doublings(
                bases,
                stop,
                range_offset=CACHE_RANGE_OFFSET,
            )

            for value, source_index in zip(
                values,
                source_indices,
            ):
                value = int(value)
                r = source_values[int(source_index)]

                if value in previous:
                    continue

                removed.add(value)
                colors.setdefault(value, color_for(r))

            return removed, colors

        bases = []
        source_values = []

        for prev in sorted(previous):
            max =1e10
            if prev > max:
                print(f"Skipping {prev} because it exceeds the max value {max}")
                continue

            r = prev

            for _ in range(depth - 2):
                r_2_coord = get_p2_index(r)
                projected = r // (1 << r_2_coord)
                next_value = projected * 3 + 1

                next_2_coord = get_p2_index(next_value)
            

                r = next_value // (1 << next_2_coord)

            numerator = prev - 1

            if numerator % 3 != 0:
                continue

            base = numerator // 3

            if base > 0 and base % 2 == 1:
                bases.append(base)
                source_values.append(r)

        values, source_indices = expand_doublings(
            bases,
            stop,
            range_offset=CACHE_RANGE_OFFSET,
        )

        for value, source_index in zip(
            values,
            source_indices,
        ):
            value = int(value)
            r = source_values[int(source_index)]

            if value in previous:
                continue

            removed.add(value)
            colors.setdefault(value, color_for(r))

        return removed, colors

    def render_levels(index, start, stop):
        original = np.arange(
            start,
            stop + 1,
            dtype=np.int64,
        )

        face = np.full(
            (len(original), 4),
            low_opacity_black,
        )

        already_colored = set()

        for colors in color_levels[:index + 1]:
            for value, color in colors.items():
                if (
                    start <= value <= stop
                    and value not in already_colored
                ):
                    face[value - start] = color
                    already_colored.add(value)

        positions = np.column_stack(
            (
                original,
                np.zeros(len(original)),
            )
        )

        return positions, face

    def filter_fn(index, x_min, x_max):
        start = max(1, int(x_min))
        stop = max(start, int(x_max))
        bounds = (start, stop)

        if bounds != cached_bounds:
            reset_cache(start, stop)

        while len(removed_levels) <= index:
            depth = len(removed_levels)

            removed, colors = calculate_level(
                depth,
                stop,
            )

            removed_levels.append(removed)
            color_levels.append(colors)

            visible_removed = {
                value
                for value in removed
                if 1 <= value <= stop
            }

            remaining.difference_update(visible_removed)

            if depth >= 3 and visible_removed:
                min_removed = min(visible_removed)
                min_remaining = min(
                    remaining,
                    default=stop + 1,
                )

                for k in range(
                    1,
                    13,
                ):
                    upper_bound = 1 << k

                    if (
                        min_removed <= upper_bound
                        and min_remaining >= upper_bound
                        and upper_bound not in reported_clears
                    ):
                        print(
                            f"Cleared (0,{upper_bound}) "
                            f"at depth {depth}"
                        )
                        reported_clears.add(upper_bound)

        return render_levels(
            index,
            start,
            stop,
        )

    def hover_fn(i, data):
        n = int(data[i, 0])

        if n > Monzo.get_biggest_loaded_prime():
            return str(n)

        return get_monzo_desc(Monzo.from_int(n))

    draw_collatz_filter(filter_fn, hover_fn)

distance_cache = {}

def direct_compute_distance(i):
    i = project_above_p2(i)
    if i <= 1:
        return 0
    if i in distance_cache:
        return distance_cache[i]

    nxt = project_above_p2(i * 3 + 1)
    result = 1 + direct_compute_distance(nxt)
    distance_cache[i] = result
    return result

max_n = 2 ** 13 + 1

for i in range(1, max_n):
    direct_compute_distance(i)

distance_cache_df = pd.DataFrame(list(distance_cache.items()), columns=["int", "distance"])

distance_cache_df = distance_cache_df[distance_cache_df["int"] < max_n]

distance_cache_df.to_csv("distance_cache.csv", index=False)

# pts = distance_cache_df[["int", "distance"]].to_numpy(dtype=float)
# hull = ConvexHull(pts)
# verts = pts[hull.vertices]

# upper = []
# n = len(verts)
# for i in range(n):
#     a, b = verts[i], verts[(i + 1) % n]
#     if a[0] != b[0] and (a[1] + b[1]) / 2 >= np.interp(
#         (a[0] + b[0]) / 2,
#         pts[np.argsort(pts[:, 0]), 0],
#         pts[np.argsort(pts[:, 0]), 1],
#     ):
#         upper.extend([a, b])

# envelope = pd.DataFrame(np.unique(upper, axis=0), columns=["int", "distance"])
# envelope.to_csv("envelope.csv", index=False)