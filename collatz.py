from vispy.color import ColorArray
from monzo import Monzo, PRIMES
import networkx as nx
from visualization import draw_collatz_graph
import numpy as np
from vispy import app
import time

def pos_mod_coords_to_string(m: Monzo):
    m = "[" + ", ".join(str(k % PRIMES[j]) for j, k in enumerate(m.get_modular_coordinates())) + "]"
    if len(m) > 50:
        m = m[:25] + "..." + m[-25:]

    return m

def print_pos_mod_coords(m: Monzo):
    print(pos_mod_coords_to_string(m))

G = nx.DiGraph()

mod_2_list = []
start_nodes = []
edges = []

def do_collatz(int):
    start_nodes.append(int)
    m = Monzo.from_int(int)

    while True:
        deux = m.get_index(0)
        m -= Monzo([deux])

        new_int = m.to_int()

        if deux > 0:
            edges.append((int, new_int))
            if new_int in mod_2_list:
                break
            else:
                mod_2_list.append(new_int)
        
        if new_int == 1:
            break

        int = new_int

        new_int = int * 3 + 1
        edges.append((int, new_int))
        
        m = Monzo.from_int(new_int)

        int = new_int

t = time.time()
for i in range(100):
    # m = Monzo.from_int(3 * i + 1)
    m = Monzo.from_int(Monzo.get_prime_of_index(i))
    do_collatz(m.to_int())

print(f"Collatz completed, size of graph: {len(start_nodes) + len(mod_2_list)} nodes, {len(edges)} edges")
print(f"Time taken: {time.time() - t}s")
t = time.time()

G.add_edges_from(edges)
G.graph["rankdir"] = "BT" 
G.graph["nodesep"] = 1.0
G.graph["ranksep"] = 0.8

pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  
# pos = nx.nx_agraph.graphviz_layout(G.reverse(), prog="twopi", root=1)

nodes = list(G.nodes())
index = {n: i for i, n in enumerate(nodes)}
xy = np.array([pos[n] for n in nodes], dtype=float)

face = np.full((len(nodes), 4), 0.5) 

odd_teal   = ColorArray("#6ca6d9").rgba
even_Green  = ColorArray("#4aa63d").rgba
evil_sable  = ColorArray("#666666").rgba
prime_pure_white = ColorArray("#FFFFFF").rgba

mod2 = set(mod_2_list)
starts = set(start_nodes)

t = time.time()
for i, n in enumerate(nodes):
    if n in starts:
        face[i] = evil_sable
    elif Monzo.is_prime(n):
        face[i] = prime_pure_white
    elif n in mod2:
        face[i] = odd_teal
    else:
        face[i] = even_Green

def label_fn(i):
    monzo = Monzo.from_int(nodes[i])
    mod_coords_str = "[" + ", ".join(str(j) for j in monzo.get_modular_coordinates()) + "]"
    if len(mod_coords_str) > 50:
        mod_coords_str = mod_coords_str[:25] + "..." + mod_coords_str[-25:]

    coords_str = "‹" + ", ".join([str(c) for c in monzo.get_coordinates()]) + "›"

    if len(coords_str) > 50:
        coords_str = coords_str[:25] + "..." + coords_str[-25:]

    return f"‖m‖²: {monzo.square_norm()}\n|m_P|: {len(monzo.get_modular_coordinates())}\nm_P(+): {pos_mod_coords_to_string(monzo)}\nm_P(-): {mod_coords_str}\nm:{coords_str}\nm_Z: {monzo.to_int()}"

print(f"Pre-graph took ({time.time() - t}s)")

canvas = draw_collatz_graph(edges, xy, face, index, label_fn=label_fn)
app.run()