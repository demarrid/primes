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
    while n != 1 and n != 16 and n > 0:
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

# reverse_collatz(1, 3, 7)

max = 3

for a in range(max):
    for b in range(max):
        for c in range(max):
            for d in range(max):
                m = Monzo.from_modular_coordinates([1, a + 1, b + 1, c + 1, d + 1])
                if m.to_int() > 1:
                    print(f"Collatz {m.to_int()}")
                    do_collatz(m.to_int())

print(f"Collatz completed, size of graph: {len(collatz_nodes) + len(predecessor_nodes)} nodes, {len(collatz_edges)} edges")
print(f"Time taken: {time.time() - t}s")
t = time.time()

G.add_edges_from(collatz_edges)
G.graph["rankdir"] = "BT" 
G.graph["nodesep"] = 1.0
G.graph["overlap_scaling"] = 2
G.graph["ranksep"] = "2.0"

pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  
# pos = nx.nx_agraph.graphviz_layout(G.reverse(), prog="twopi", root=1)
# pos = nx.nx_agraph.graphviz_layout(G, prog="neato") 

nodes = list(G.nodes())
index = {n: i for i, n in enumerate(nodes)}
xy = np.array([pos[n] for n in nodes], dtype=float)

face = np.full((len(nodes), 4), 0.5) 

odd_teal   = ColorArray("#6ca6d9").rgba
evil_sable  = ColorArray("#666666").rgba
source  = ColorArray("#edb193").rgba
pure_source = ColorArray("#FFFFFF").rgba
green_goblin = ColorArray("#1ced1c").rgba
orange_obviant = ColorArray("#DB782A").rgba

print(f"Graph organization took ({time.time() - t}s)")

t = time.time()
for i, n in enumerate(nodes):
    if n in source_nodes:
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
    mod_coords_str = "[" + ", ".join(str(j) for j in monzo.get_modular_coordinates()) + "]"
    if len(mod_coords_str) > 50:
        mod_coords_str = mod_coords_str[:25] + "..." + mod_coords_str[-25:]

    coords_str = "‹" + ", ".join([str(c) for c in monzo.get_coordinates()]) + "›"

    if len(coords_str) > 50:
        coords_str = coords_str[:25] + "..." + coords_str[-25:]

    return f"‖m‖²: {monzo.square_norm()}\n|m_P|: {len(monzo)}\nm_P(+): {pos_mod_coords_to_string(monzo)}\nm_P(-): {mod_coords_str}\nm:{coords_str}\nm_Z: {monzo.to_int()}"

print(f"Pre-graph took ({time.time() - t}s)")

canvas = draw_collatz_graph(collatz_edges, xy, face, index, label_fn=label_fn)
app.run()