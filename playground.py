from monzo import Monzo, get_monzo
from utils import render_grid
import pandas as pd

monzos = [get_monzo(i) for i in range(1, 10000)]

monzo_grid = [[get_monzo(monzos[i].get_index(j)).as_color() for j in range(len(monzos[i]))] for i in range(len(monzos))]
columns = [f"color{i+1}" for i in range(16)]
render_grid(monzo_grid, path="graphs/monzo_grid.png")

grid_df = pd.DataFrame(columns=["int", *columns])

for mz in monzos:
    for i in range(len(mz)):
        pow = mz.get_index(i)
        if pow > 0:
            grid_df = pd.concat([grid_df, pd.DataFrame([{"int": mz.to_int(), **{columns[pow - 1]: Monzo.get_nth_prime(i + 1)}}])])

grid_df.to_csv("grid_df.csv", index=False)

# Precompute all square norms only once
square_norms = [mz.square_norm() for mz in monzos]
square_norm_monzos = [get_monzo(sq) for sq in square_norms]

# square_norm_grid = [
#     [square_norm_monzos[i].get_index(j) for j in range(len(monzos[i]))]
#     for i in range(len(monzos))
# ]
# square_norm_grid = [
#     [get_monzo(val).as_color() for val in row]
#     for row in square_norm_grid
# ]
# render_grid(square_norm_grid, path="graphs/square_norm_grid.png")

square_norm_df = pd.DataFrame(columns=["int", *columns])

for mz, nmz in zip(monzos, square_norm_monzos):
    for i in range(len(nmz)):
        pow = nmz.get_index(i)
        if pow > 0:
            square_norm_df = pd.concat([square_norm_df, pd.DataFrame([{"int": mz.to_int(), **{columns[pow - 1]: Monzo.get_nth_prime(i + 1)}}])])

square_norm_df.to_csv("square_norm_df.csv", index=False)

normalized_df = pd.DataFrame(columns=["int", "normalized_val"])

for i, mz in enumerate(monzos):
    normalized_df = pd.concat([normalized_df, pd.DataFrame([{"int": mz.to_int(), "normalized_val": (mz / max(mz.norm(), 1)).to_fraction()}])])

normalized_df.to_csv("normalized_monzos.csv", index=False)

# odd primes are x^2 = y^2

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