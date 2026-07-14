from vispy.color import ColorArray
from monzo import Monzo
import networkx as nx
from visualization import draw_collatz_graph
import numpy as np
from vispy import app
import time

from utils import pos_mod_coords_to_string

def print_pos_mod_coords(m: Monzo):
    print(pos_mod_coords_to_string(m))

G = nx.DiGraph()

mod_2_nodes = []
start_nodes = []
edges = []
predecessor_nodes = []
source_nodes = []

def do_collatz(n):
    copy = n
    while n > 0 and n % 2 == 0:
        n //= 2
    while n != 1:
        nxt = 3 * n + 1
        while nxt % 2 == 0:
            nxt //= 2
        edges.append((n, nxt))

        if n == 43079:
            print(f"n: {n}\nnxt: {nxt} \ncopy: {copy}")
        
        if nxt in mod_2_nodes:
            break
        mod_2_nodes.append(nxt)
        n = nxt

t = time.time()

def predecessors_of(m: int, count: int):
    while m > 0 and m % 2 == 0:
        m //= 2

    k = m % 3 

    if k == 0:
        source_nodes.append(m)
        return []

    to_return = []

    for j in range(1, count + 1):
        new_2_index = 2*(j+1) + k - 1
        value_with_new_2_index = m * (2**new_2_index)
        z = (value_with_new_2_index - 1) // 3

        to_return.append(int(z))
        
    return to_return

def reverse_collatz(sink:int, width:int, depth:int):
    if sink % 3 == 0:
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
        if len(preds) < width:
            print(f"Insufficient predecessors for {sink} at depth {depth} ({len(preds)} < {width})")

reverse_collatz(1, 50, 2)

for i in range(2000):
   do_collatz(i)

print(f"Collatz completed, size of graph: {len(mod_2_nodes) + len(predecessor_nodes)} nodes, {len(edges)} edges")
print(f"Time taken: {time.time() - t}s")
t = time.time()

G.add_edges_from(edges)
G.graph["rankdir"] = "BT" 
G.graph["nodesep"] = 1.0
G.graph["ranksep"] = 0.8

# pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  
pos = nx.nx_agraph.graphviz_layout(G.reverse(), prog="twopi", root=1)

nodes = list(G.nodes())
index = {n: i for i, n in enumerate(nodes)}
xy = np.array([pos[n] for n in nodes], dtype=float)

face = np.full((len(nodes), 4), 0.5) 

odd_teal   = ColorArray("#6ca6d9").rgba
evil_sable  = ColorArray("#666666").rgba
source  = ColorArray("#edb193").rgba
pure_source = ColorArray("#FFFFFF").rgba
green_goblin = ColorArray("#1ced1c").rgba

t = time.time()
for i, n in enumerate(nodes):
    if n in source_nodes:
        face[i] = pure_source
    elif n in start_nodes:
        face[i] = evil_sable
    elif n in predecessor_nodes:
        face[i] = green_goblin
    elif n in mod_2_nodes:
        face[i] = odd_teal

def label_fn(i):
    if i > 1e12:
        return str(nodes[i])

    monzo = Monzo.from_int(nodes[i])
    mod_coords_str = "[" + ", ".join(str(j) for j in monzo.get_modular_coordinates()) + "]"
    if len(mod_coords_str) > 50:
        mod_coords_str = mod_coords_str[:25] + "..." + mod_coords_str[-25:]

    coords_str = "‹" + ", ".join([str(c) for c in monzo.get_coordinates()]) + "›"

    if len(coords_str) > 50:
        coords_str = coords_str[:25] + "..." + coords_str[-25:]

    return f"‖m‖²: {monzo.square_norm()}\n|m_P|: {len(monzo)}\nm_P(+): {pos_mod_coords_to_string(monzo)}\nm_P(-): {mod_coords_str}\nm:{coords_str}\nm_Z: {monzo.to_int()}"

print(f"Pre-graph took ({time.time() - t}s)")

canvas = draw_collatz_graph(edges, xy, face, index, label_fn=label_fn)
app.run()