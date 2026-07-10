from monzo import Monzo, target_max_int
import pandas as pd
from monzo import PRIMES
import visualization as viz
from utils import factor, make_spf, build_grid_records, load_or_build, build_modular_coord_records
import numpy as np
from vispy import app

N = target_max_int
visualize = False
if visualize:
    monzos = [Monzo.get(i) for i in range(1, N + 1)]

    spf = make_spf(N)
    prime_index = {p: i for i, p in enumerate(PRIMES)}

    grid_df = load_or_build(
        f"grid_df_{N}.csv",
        lambda: build_grid_records(N, spf, prime_index),
    )

    sqnorm_df = load_or_build(
        f"square_norm_df_{N}.csv",
        lambda: build_grid_records(N, spf, prime_index,
                                value_of=lambda n, exps: sum(e*e for e in exps.values())),
    )

    def build_normalized_records(N, spf):
        rows = []
        for n in range(2, N + 1):
            exps = list(factor(n, spf))           
            norm = np.sqrt(sum(e*e for _, e in exps)) or 1
            val = 1.0
            for p, e in exps:
                val *= p ** (e / norm)
            rows.append((n, val))
        return pd.DataFrame(rows, columns=["int", "normalized_val"])

    normalized_df = load_or_build(f"normalized_monzos_{N}.csv",
                            lambda: build_normalized_records(N, spf))

    modular_coords_df = load_or_build(
        f"modular_coords_df_{N}.csv",
        lambda: build_modular_coord_records(N, spf, prime_index),
    )

    canvases = [
        # viz.scatter_view(grid_df, "int", "prime_index", value_col="exponent", title="grid", line_segments=False),
        # viz.scatter_view(sqnorm_df, "int", "prime_index", value_col="exponent", title="square norm"),
        # viz.scatter_view(normalized_df, "int", "normalized_val", value_col="normalized_val", continuous=True, title="normalized"),
        viz.scatter_view(modular_coords_df, "int", "prime_index", value_col="coord", title="modular coordinates",),
    ]
    app.run()

# odd primes are x^2 + y^2

# goldbach: every even integer is the sum of two primes
# twin prime: there are infinitely many pairs of primes that differ by 2
# legendre: there is always at least one prime between consecutive perfect squares
# there are infinitely many primes of the form n^2 + 1

# balanced primes: arithmetic mean of their next larger and next smaller prime
# bell primes: every prime x so there exists a finite set for which the total amount of partitions is x
# chen primes: primes p so p+2 is either prime or semiprime (euclidean square norm at most 2)
# cluster primes: prime p with every natural number k less than p- 3 is the difference of two primes at most p
# cuban: p=k^3-(k-1)^3
# cullen: p=n*2^n+1
# eisenstein: 3k-1
# euclid: p-1 is the product of the first k primes for some k
# factorial primes: p=n!\pm 1
# fermat: p=2^(2^n)+1
# fibonacci: p=F_n for some n
# good primes: ab < p^2 for all primes a, b < p
# irregular: odd p that divides class number of p-th cyclotomic field
# leyland: p=a^b+b^a, a,b>1
# non-generous primes: p, the least positive primiive root is not a primitive root of p^2
# safe-prime: p and (p-1)/2 are both prime
# super-prime: p is the kth prime, k is prime

# for m in range(1, 7):
#     d = Monzo.from_modular_coordinates([-1, -1, -1, -m])
#     print(d)
#     print(d.get_modular_coordinates())
#     print(d.to_int())
#     e = Monzo.from_modular_coordinates([-1, -2, 1, -m + 2])
#     print(e)
#     print(e.get_modular_coordinates())
#     print(e.to_int())

funny = 13
m = Monzo.from_int(funny)

for i in range (1, 10):
    print(f"====== Step {i} =======")
    print(m)
    print(m.get_modular_coordinates())
    print(m.to_int())
    if m.get_index(0) > 0:
        m = m - Monzo([1])
    else:
        m = (m + Monzo([0,1])).successor()
    