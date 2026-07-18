from monzo import Monzo, PRIMES
import networkx as nx
from visualization import draw_goldbach_graph
import numpy as np
from vispy import app
import time
from vispy.color import ColorArray
from utils import pos_mod_coords_to_string
from vispy.color import get_colormap
t = time.time()
even_nodes = []
prime_nodes = []
edges = []
even_to_pair = {}

for i in range(4, 10000, 2):
    m = Monzo.from_int(i)
    int_val = m.to_int()
    pair = None
    left = 0
    right = len(PRIMES) - 1
    while left <= right and PRIMES[left] <= int_val // 2:
        p1 = PRIMES[left]
        p2 = int_val - p1
        lo, hi = left, right
        while lo <= hi:
            mid = (lo + hi) // 2
            if PRIMES[mid] == p2:
                pair = (p1, p2)
                break
            elif PRIMES[mid] < p2:
                lo = mid + 1
            else:
                hi = mid - 1
        if pair is not None:
            break
        left += 1

    # cognitive offloading i haven't had a migraine in like a week this is the first time :thumbsup:
    
    even_nodes.append(int_val)
    if pair[0] not in prime_nodes:
        prime_nodes.append(pair[0])
    if pair[1] not in prime_nodes:
        prime_nodes.append(pair[1])

    edges.append((int_val, pair[0]))
    edges.append((int_val, pair[1]))
    even_to_pair[int_val] = pair
    
    #it's 2 4 6 and 1 3 5

print(f"Goldbach completed, size of graph: {len(even_nodes) + len(prime_nodes)} nodes, {len(edges)} edges")
print(f"Time taken: {time.time() - t}s")
t = time.time()

P = nx.Graph()
P.add_nodes_from(prime_nodes)

for edge, (p1, p2) in even_to_pair.items():
    P.add_edge(p1, p2)

pos = nx.circular_layout(P)

for e, (p1, p2) in even_to_pair.items():
    pos[e] = (np.array(pos[p1]) + np.array(pos[p2])) / 2

nodes = list(pos.keys())
index = {n: i for i, n in enumerate(nodes)}
xy = np.array([pos[n] for n in nodes], dtype=float)

face = np.full((len(nodes), 4), 0.5) 

odd_teal   = ColorArray("#6ca6d9").rgba
even_Green  = ColorArray("#4aa63d").rgba
evil_sable  = ColorArray("#666666").rgba
prime_pure_white = ColorArray("#FFFFFF").rgba

primes = set(prime_nodes)
evens = set(even_nodes)

t = time.time()
vals = np.log1p(np.array(nodes, dtype=float))
norm = (vals - vals.min()) / (np.ptp(vals) or 1)
face = get_colormap("viridis").map(norm)

def label_fn(i, data):
    monzo = Monzo.from_int(nodes[i])
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