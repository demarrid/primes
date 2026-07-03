from monzo import Monzo, target_max_int
import pandas as pd
from monzo import PRIMES
import visualization as viz
from utils import factor, make_spf, build_grid_records, load_or_build
import numpy as np
from vispy import app

N = target_max_int

# monzos = [Monzo.get(i) for i in range(1, N + 1)]

# monzo_grid = [[Monzo.get(monzos[i].get_index(j)).as_color() for j in range(len(monzos[i]))] for i in range(len(monzos))]
# columns = [f"color{i+1}" for i in range(16)]
# render_grid(monzo_grid, path=f"graphs/monzo_grid_{upper_bound}.png")

# square_norms = [mz.square_norm() for mz in monzos]
# square_norm_monzos = [Monzo.get(sq) for sq in square_norms]

# square_norm_grid = [
#     [square_norm_monzos[i].get_index(j) for j in range(len(monzos[i]))]
#     for i in range(len(monzos))
# ]

# square_norm_grid = [
#     [Monzo.get(val).as_color() for val in row]
#     for row in square_norm_grid
# ]

# render_grid(square_norm_grid, path=f"graphs/square_norm_grid_{upper_bound}.png")

# square_norm_df = pd.DataFrame(columns=["int", *columns])

# for mz, nmz in zip(monzos, square_norm_monzos):
#     for i in range(len(nmz)):
#         pow = nmz.get_index(i)
#         if pow > 0:
#             square_norm_df = pd.concat([square_norm_df, pd.DataFrame([{"int": mz.to_int(), **{columns[pow - 1]: Monzo.get_nth_prime(i + 1)}}])])

# square_norm_df.to_csv(f"square_norm_df_{upper_bound}.csv", index=False)

# normalized_df = pd.DataFrame(columns=["int", "normalized_val"])

# for i, mz in enumerate(monzos):
#     normalized_df = pd.concat([normalized_df, pd.DataFrame([{"int": mz.to_int(), "normalized_val": (mz / max(mz.norm(), 1)).to_fraction()}])])

# normalized_df.to_csv(f"normalized_monzos_{N}.csv", index=False)

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
canvases = [
    # viz.scatter_view(grid_df, "int", "prime_index", value_col="exponent", title="grid", line_segments=True),
    viz.scatter_view(sqnorm_df, "int", "prime_index", value_col="exponent", title="square norm"),
    # viz.scatter_view(normalized_df, "int", "normalized_val", value_col="normalized_val", continuous=True, title="normalized"),
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

# what is a successor? 
# we have an element of the vector space v = (c_1, c_2, c_3, ...)
# primes are the orthonormal basis vectors e_i
# we have some int k = prod_{i=1}^\infty e_i^c_i
# its norm squared is sum_{i=1}^\infty c_i^2
# k's successor s must have (prod_{i=1}^\infty e_i^{a_i}) - (prod_{i=1}^\infty e^{c_i}) = 1
# we know <v, s> = 0 hence c_i > 0 \implies a_i = 0 (s is in the orthogonal complement of the span of v's nonzero coordinates)
# we must minimize |(prod_{i=i}^\infty e_i^{a_i}) - (prod_{i=i}^\infty e^{c_i})|
# i.e., minimize |\sum{i=i}^\infty a_i\log(e_i) - \sum{i=i}^\infty c_i\log(e_i)|
# i.e., minimize |\sum{i=1}^\infty a_i * i - \sum{i=1}^\infty c_i * i|
# since we only take positive values, use dynamic programming ?

# try: inductively increment coordinates from e_k down to e_1 while maintaining condition
# try: define inner product by multiplying by p or logp for each coordinate
funny_integer = 17
v = Monzo.get(funny_integer)
valid_indices = [i for i in range(len(v)) if v.get_index(i) == 0 and PRIMES[i] < funny_integer]

left_hand_sum = sum(v.get_index(i) * np.log(PRIMES[i]) for i in range(len(v)))
# dirichlet: if a and b are coprime, then there are infinitely many primes of the form an + b
remaining_diff = left_hand_sum

print("Value of LHS:", left_hand_sum)

empty_array = [0] * len(v)

if funny_integer % 2 == 1:
    print("Successor should be even")
    empty_array[0] = 1
    remaining_diff -= np.log(2)
    print("post-even adjustment remaining_diff: ", remaining_diff)

while (len(valid_indices) > 0):
    last_list_index = len(valid_indices) - 1
    coordinate_index = valid_indices[last_list_index]
    p = PRIMES[coordinate_index]
    prime_log = np.log(p)
    k = 1
    print("Coordinate Index: ", coordinate_index)
    print("Prime: ", p)
    print("Prime log: ", prime_log)
    print("Value of k: ", k)
    theoretical_diff = remaining_diff - k * prime_log
    if (theoretical_diff < 0):
        if p == 2 and 2 * remaining_diff > prime_log:
            print("Last prime has remaining diff, increasing")
            k += 1
            remaining_diff -= prime_log
            empty_array[coordinate_index] = int(k)

        else:
            print("Theoretical difference is negative, breaking")
    else:
        remaining_diff = theoretical_diff
        empty_array[coordinate_index] += int(k)
    
    valid_indices.pop(last_list_index)

    print("Remaining difference: ", remaining_diff)
    print("Valid indices: ", valid_indices)
    print("Successor monzo: ", empty_array)

successor = Monzo(empty_array)
print("Successor: ", successor.to_int())
# 15 = 01100000
# 16 = 40000000
# 17 = 00000010
# 18 = 12000000
# 19 = 00000001
# 20 = 20100000

# suppose our successor is not prime


# if we are composite odd, all coordinates for primes for successor at least 1/2 of us are 0
# if we are even, all coordinates for successor at least (1/3 of us + our highest coordinate) value are 0