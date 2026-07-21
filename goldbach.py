from monzo import Monzo, PRIMES
import networkx as nx
from visualization import draw_goldbach_graph
import numpy as np
from vispy import app
import time
from vispy.color import ColorArray
from utils import pos_mod_coords_to_string
from vispy.color import get_colormap
import pandas as pd
t = time.time()
even_nodes = []
prime_nodes = []
edges = []
even_to_primes = []

def calc_graph():
    for i in range(4, 1000, 2):
        m = Monzo.from_int(i)
        int_val = m.to_int()
        left = 0
        right = len(PRIMES) - 1
        possible_p2 = []
        while left <= right and PRIMES[left] <= int_val // 2:
            p1 = PRIMES[left]
            p2 = int_val - p1
            lo, hi = left, right
            while lo <= hi:
                mid = (lo + hi) // 2
                if PRIMES[mid] == p2:
                    possible_p2.append(mid)
                    lo = mid + 1

                elif PRIMES[mid] < p2:
                    lo = mid + 1
                else:
                    hi = mid - 1
            left += 1
        
        for p2_index in possible_p2:
            p2 = PRIMES[p2_index]
            p1 = int_val - p2
            # cognitive offloading i haven't had a migraine in like a week this is the first time :thumbsup:
            even_nodes.append(int_val)
            if p1 not in prime_nodes:
                prime_nodes.append(p1)
            if p2 not in prime_nodes:
                prime_nodes.append(p2)
            edges.append((int_val, p1))
            edges.append((int_val, p2))

            # print( Monzo.from_int(p2).get_modular_coordinates())
            even_to_primes.append((i, p1, p2))

    index_to_primes_df = pd.DataFrame([
        {'index': even, 'p1': p1, 'p2': p2}
        for even, p1, p2 in even_to_primes
    ])
    index_to_primes_df.to_csv('index_to_primes.csv', index=False)

    print(f"Goldbach completed, size of graph: {len(even_nodes) + len(prime_nodes)} nodes, {len(edges)} edges")
    print(f"Time taken: {time.time() - t}s")

def draw_graph():
    t = time.time()

    P = nx.Graph()
    P.add_nodes_from(prime_nodes)

    for even, p1, p2 in even_to_primes:
        P.add_edge(p1, p2)

    pos = nx.circular_layout(P)

    # synthetic ids that cannot collide with prime values
    base = max(prime_nodes) + 1
    edges = []
    even_by_node = {}

    for i, (even, p1, p2) in enumerate(even_to_primes):
        pair_id = base + i
        pos[pair_id] = (np.array(pos[p1]) + np.array(pos[p2])) / 2
        edges.append((pair_id, p1))
        edges.append((pair_id, p2))
        even_by_node[pair_id] = even

    nodes = list(pos.keys())
    index = {n: i for i, n in enumerate(nodes)}
    xy = np.array([pos[n] for n in nodes], dtype=float)

    t = time.time()
    # color by the underlying integer (even or prime), not the synthetic id
    vals = np.log1p(np.array(
        [even_by_node.get(n, n) for n in nodes],
        dtype=float,
    ))
    norm = (vals - vals.min()) / (np.ptp(vals) or 1)
    face = get_colormap("viridis").map(norm)

    def label_fn(i, data):
        n = nodes[i]
        monzo = Monzo.from_int(even_by_node.get(n, n))
        mod_coords_str = "[" + ", ".join(str(j) for j in monzo.get_modular_coordinates()) + "]"
        if len(mod_coords_str) > 50:
            mod_coords_str = mod_coords_str[:25] + "..." + mod_coords_str[-25:]

        coords_str = "‹" + ", ".join([str(c) for c in monzo.get_coordinates()]) + "›"
        if len(coords_str) > 50:
            coords_str = coords_str[:25] + "..." + coords_str[-25:]

        return f"‖m‖²: {monzo.square_norm()}\n|m_P|: {len(monzo)}\nm_P(+): {pos_mod_coords_to_string(monzo)}\nm_P(-): {mod_coords_str}\nm:{coords_str}\nm_Z: {monzo.to_int()}"

    print(f"Pre-graph took ({time.time() - t}s)")

    draw_goldbach_graph(edges, xy, face, index, label_fn=label_fn)
    app.run()


calc_graph()
draw_graph()