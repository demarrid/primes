from monzo import PRIMES
import pandas as pd
import visualization as viz
import numpy as np
from vispy import app
from clanker_utils import make_spf, factor, load_or_build

# i in 0..I_MAX (even only by default — odd i give even i^2+1)
I_MAX = 200
# modular coords are O(π(i^2)) per i, so keep this smaller
I_MOD = 400
EVEN_ONLY = True
visualize = True


def almost_square_params(I_MAX, even_only=True):
    step = 2 if even_only else 1
    for i in range(0, I_MAX + 1, step):
        yield i, i * i + 1


def build_as_grid_records(I_MAX, spf, prime_index, even_only=True,
                          value_of=lambda i, n, exps: n):
    records = []
    for i, n in almost_square_params(I_MAX, even_only):
        if n < 2:
            continue
        exps = dict(factor(n, spf))
        target = value_of(i, n, exps)
        if target < 2:
            continue
        for p, e in factor(target, spf):
            records.append((i, n, prime_index[p], p, e))
    return pd.DataFrame(
        records, columns=["int", "value", "prime_index", "prime", "exponent"]
    )


def build_as_normalized_records(I_MAX, spf, even_only=True):
    rows = []
    for i, n in almost_square_params(I_MAX, even_only):
        if n < 2:
            continue
        exps = list(factor(n, spf))
        norm = np.sqrt(sum(e * e for _, e in exps)) or 1
        val = 1.0
        for p, e in exps:
            val *= p ** (e / norm)
        rows.append((i, n, val))
    return pd.DataFrame(rows, columns=["int", "value", "normalized_val"])


def build_as_modular_coord_records(I_MAX, primes, even_only=True):
    # primes must be sorted ascending
    records = []
    for i, n in almost_square_params(I_MAX, even_only):
        if n < 2:
            continue
        for idx, p in enumerate(primes):
            if p > n:
                break
            records.append((i, n, idx, (n % p) / float(p)))
    return pd.DataFrame(
        records, columns=["int", "value", "prime_index", "coord"]
    )


if visualize:
    max_n = I_MAX * I_MAX + 1
    spf = make_spf(max_n)
    primes = [p for p in PRIMES if p <= max_n]
    prime_index = {p: idx for idx, p in enumerate(primes)}

    tag = f"as_{I_MAX}{'_even' if EVEN_ONLY else ''}"
    mod_tag = f"as_{I_MOD}{'_even' if EVEN_ONLY else ''}"

    grid_df = load_or_build(
        f"grid_df_{tag}.csv",
        lambda: build_as_grid_records(I_MAX, spf, prime_index, EVEN_ONLY),
    )

    sqnorm_df = load_or_build(
        f"square_norm_df_{tag}.csv",
        lambda: build_as_grid_records(
            I_MAX, spf, prime_index, EVEN_ONLY,
            value_of=lambda i, n, exps: sum(e * e for e in exps.values()),
        ),
    )

    normalized_df = load_or_build(
        f"normalized_monzos_{tag}.csv",
        lambda: build_as_normalized_records(I_MAX, spf, EVEN_ONLY),
    )

    modular_coords_df = load_or_build(
        f"modular_coords_df_{mod_tag}.csv",
        lambda: build_as_modular_coord_records(I_MOD, primes, EVEN_ONLY),
    )

    canvases = [
        viz.scatter_view(
            grid_df, "int", "prime_index", value_col="exponent",
            title="i^2+1 grid", line_segments=False,
        ),
        viz.scatter_view(
            sqnorm_df, "int", "prime_index", value_col="exponent",
            title="i^2+1 square norm",
        ),
        viz.scatter_view(
            normalized_df, "int", "normalized_val", value_col="normalized_val",
            continuous=True, title="i^2+1 normalized",
        ),
        viz.scatter_view(
            modular_coords_df, "int", "prime_index", value_col="coord",
            title="i^2+1 modular coordinates",
        ),
    ]
    app.run()
