import os
import importlib
from functools import lru_cache
from monzo import PRIMES
import numpy as np
import pandas as pd

@lru_cache(maxsize=1)
def _cupy():
    """Return CuPy when it is installed and can actually run a kernel.

    Detecting a device is not enough: CuPy may still fail to compile kernels
    at call time when the CUDA toolkit headers are missing. We probe with a
    tiny kernel so those environments fall back to NumPy instead of crashing.
    """
    try:
        cp = importlib.import_module("cupy")
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return None
        # Force a real kernel launch to verify the toolchain is usable.
        int(cp.arange(1, dtype=cp.int64).sum())
        return cp
    except Exception:
        return None

def gpu_available():
    """Whether CuPy can currently access a GPU."""
    return _cupy() is not None

def expand_doublings(
    bases,
    x_max,
    range_offset=0,
    prefer_gpu=True,
    gpu_threshold=100_000,
):
    """Expand positive bases by powers of two in one vectorized batch.

    For each base, this reproduces:

        range(int(log2(x_max / base)) + range_offset)

    The returned ``source_indices`` identifies the input base that produced
    each value. GPU acceleration is used only when CuPy is available and the
    temporary matrix is large enough to justify transfer overhead.
    """
    bases = np.asarray(list(bases), dtype=np.int64)
    if bases.ndim != 1:
        raise ValueError("bases must be one-dimensional")
    if bases.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if np.any(bases <= 0) or x_max <= 0:
        raise ValueError("bases and x_max must be positive")

    counts = np.floor(np.log2(x_max / bases)).astype(np.int64) + int(range_offset)
    counts = np.maximum(counts, 0)
    width = int(counts.max(initial=0))
    if width == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    largest_values = [
        int(base) << (int(count) - 1)
        for base, count in zip(bases, counts)
        if count > 0
    ]
    if largest_values and max(largest_values) > np.iinfo(np.int64).max:
        raise OverflowError("expanded values do not fit in a signed 64-bit integer")

    cp = _cupy() if prefer_gpu and bases.size * width >= gpu_threshold else None
    xp = cp if cp is not None else np
    xp_bases = xp.asarray(bases)
    xp_counts = xp.asarray(counts)
    powers = xp.arange(width, dtype=xp.int64)
    keep = powers[None, :] < xp_counts[:, None]
    matrix = xp_bases[:, None] * xp.left_shift(xp.int64(1), powers)[None, :]
    values = matrix[keep]
    source_indices = xp.broadcast_to(
        xp.arange(bases.size, dtype=xp.int64)[:, None],
        matrix.shape,
    )[keep]

    if cp is not None:
        return cp.asnumpy(values), cp.asnumpy(source_indices)
    return values, source_indices

def load_or_build(path, build_fn, keep=True):
    if os.path.exists(path):
        return pd.read_csv(path)
    df = build_fn()
    df.to_csv(path, index=False)
    return df if keep else None  # discard from memory when not needed

def make_spf(N):
    spf = np.arange(N + 1)
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            spf[i * i :: i] = np.minimum(spf[i * i :: i], i)
    return spf

def factor(m, spf):
    """yield (prime, exponent) for m using the sieve."""
    while m > 1:
        p = int(spf[m])
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        yield p, e

def build_grid_records(N, spf, prime_index, value_of=lambda n, exps: n):
    """value_of maps (n, its exponent dict) -> the integer to factor for the grid."""
    records = []
    for n in range(2, int(N) + 1):
        exps = dict(factor(n, spf))
        target = value_of(n, exps)
        if target < 2:
            continue
        for p, e in factor(target, spf):
            records.append((n, prime_index[p], p, e))
    cols = ["int", "prime_index", "prime", "exponent"]
    return pd.DataFrame(records, columns=cols)

def build_modular_coord_records(N, spf, prime_index):
    records = []
    for n in range(1, int(N) + 1):
        max_i = 0
        for i in range(len(PRIMES)):
            if PRIMES[i] > n:
                max_i = i - 1
                break

        for i in range(max_i + 1):
            prime = PRIMES[i]
            coord = n % prime
            records.append((n, i, coord / float(prime)))
    return pd.DataFrame(records, columns=["int", "prime_index", "coord"])