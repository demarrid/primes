from vispy.color import ColorArray
from monzo import Monzo
import networkx as nx
from visualization import draw_collatz_filter, draw_collatz_graph
import numpy as np
from vispy import app
import time

from utils import pos_mod_coords_to_string

G = nx.DiGraph()

collatz_nodes = []
start_nodes = []
collatz_edges = []
predecessor_nodes = []
source_nodes = []

def improvement(n, nxt):
    n = 3 * n + 1.0
    nxt = 3 * nxt + 1.0
    n = np.log2(n) % 1
    nxt = np.log2(nxt) % 1
    return abs(nxt - n)

show_odd_nodes = False


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

    reverse_collatz(4, 12, 4)

    do_collatz(170)

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

    odd_teal = ColorArray("#6ca6d9").rgba
    evil_sable = ColorArray("#666666").rgba
    source = ColorArray("#edb193").rgba
    pure_source = ColorArray("#FFFFFF").rgba
    green_goblin = ColorArray("#1ced1c").rgba
    orange_obviant = ColorArray("#DB782A").rgba

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


    def label_fn(i):
        if nodes[i] > Monzo.get_prime_of_index(-1):
            return str(nodes[i])

        monzo = Monzo.from_int(nodes[i])
        mod_coords_str = (
            "[" + ", ".join(str(j) for j in monzo.get_modular_coordinates()) + "]"
        )
        if len(mod_coords_str) > 50:
            mod_coords_str = mod_coords_str[:25] + "..." + mod_coords_str[-25:]

        coords_str = "‹" + ", ".join([str(c) for c in monzo.get_coordinates()]) + "›"

        if len(coords_str) > 50:
            coords_str = coords_str[:25] + "..." + coords_str[-25:]

        return f"‖m‖²: {monzo.square_norm()}\n|m_P|: {len(monzo)}\nm_P(+): {pos_mod_coords_to_string(monzo)}\nm_P(-): {mod_coords_str}\nm:{coords_str}\nm_Z: {monzo.to_int()}"


    print(f"Pre-graph took ({time.time() - t}s)")

    draw_collatz_graph(collatz_edges, xy, face, index, label_fn=label_fn)
    app.run()

def filter():
    def filter_fn(index, x_min, x_max):
        x_min = max(1, x_min)
        result = [i for i in range(int(x_min), int(x_max) + 1)]

        removed_0 = []
        if index >= 0:
            for r in range(1, int(np.log2(x_max)) + 1):
                result.remove(2 ** r)
                removed_0.append(2 ** r)
                # print(f"Removed 0: {2 ** i}")

        removed_1 = []
        if index >= 1:
            for r in removed_0:
                r = (r - 1) / 3
                if r.is_integer():
                    for l in range(0, int(2* np.log2( x_max)) + 1):
                        v = int(r * 2 ** l)
                        if v in result:
                            result.remove(v)
                            removed_1.append(v)
                            # print(f"Removed 1: {v}")

        removed_2 = []

        if index >= 2:
            for r in removed_1:
                if r % 3 != 2:
                    continue
                for p in range(0, int(x_max / r) + 1):
                    if p % r == 0 or r % p == 0:
                        continue
                    possible_z = ((2**p) * r - 1) / 3
                    if possible_z.is_integer() and possible_z > 0 and possible_z % 2 == 1:
                        for power in range(0, int(2 *np.log2( x_max)) + 1):
                            v = int(possible_z * 2 ** power)
                            if v in result:
                                result.remove(v)
                                removed_2.append(v)
                                # print(f"Removed 2: {v}")
                                if v == 23:
                                    print(f"v: {v}")
                                    print(f"possible_z: {possible_z}")
                                    print(f"power: {power}")
                                    print(f"p: {p}")
                                    print(f"r: {r}")
                                    

        print(f"removed_2: {removed_2}")
        removed_d = [[] for _ in range(max(0, index))]
        if index >= 3:
            for f in range(0, index - 2):

                d = f + 3
                prevs = removed_d[f - 1] if f > 0 else removed_2
                print(f"prevs for d={d}: {prevs}")
                for prev in prevs:
                    u = Monzo.from_int(prev).get_index(0)
                    w = Monzo.from_int((prev / (2**u))*3+1).get_index(0)
                    r = 3 **(d-2) * (prev / (2**u))+3**(d-3)+ 2**w


                    # this gives 2^k * 3?

                    r = ((r/3) - 1) / 3

                    print(f"d: {d}")
                    print(f"prev: {prev}")
                    print(f"u: {u}")
                    print(f"w: {w}")
                    print(f"r: {r}")

                    if r % 3 != 2 or prev % 3 == 0:
                        continue

                    x_mod_r = (1/3) * (prev - 1)
                    # print(f"x_mod_r: {x_mod_r}")
                    if x_mod_r.is_integer() and x_mod_r % 2 == 1:
                        for z in range(0, int(x_max / abs(x_mod_r)) + 1):
                            # print(f"z: {z}")
                            v = int(x_mod_r - r*z)

                            # 27 * 17 + 3 + 2^2 = 
                            if (3**(d-1) * Monzo.from_int(v).with_index(0,0).to_int() +3**(d-2) + 2**u) % r == 0:
                                # print(f"v: {v}")
                                if v in result:
                                    result.remove(v)
                                    removed_d[f].append(v)
                                    print(f"Removed {d}: {v}")

        return np.column_stack((result, np.zeros(len(result))))

    draw_collatz_filter(filter_fn)

filter()