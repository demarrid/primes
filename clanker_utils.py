import os
import importlib
from functools import lru_cache
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
    # imported lazily: monzo spends ~10s and ~1GB building PRIMES at import time
    from monzo import PRIMES

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

_SIEVE_CACHE = ".sieve_cache"
_BLOCK_SHIFT = 7
_BLOCK_ODDS = 1 << _BLOCK_SHIFT  # 128 odd slots span 256 consecutive integers

def _odd_sieve(limit):
    """Primality flags for the odd numbers up to ``limit``; slot k holds 2k+1.

    The array is padded up to a whole number of blocks so the counting tables
    below can be built with a single reshape.
    """
    n_odds = limit // 2 + 1
    sieve = np.ones(n_odds + (-n_odds % _BLOCK_ODDS), dtype=bool)
    sieve[0] = False  # 1 is not prime
    sieve[(limit - 1) // 2 + 1 :] = False  # padding, and anything past the limit
    for k in range(1, int(limit**0.5) // 2 + 1):
        if sieve[k]:
            p = 2 * k + 1
            sieve[(p * p - 1) // 2 :: p] = False
    return sieve

def _load_odd_sieve(limit, cache_dir):
    if not cache_dir:
        return _odd_sieve(limit)
    path = os.path.join(cache_dir, f"odd_sieve_{limit}.npy")
    if os.path.exists(path):
        return np.load(path)
    sieve = _odd_sieve(limit)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, sieve)
    return sieve

def _block_prime_counts(sieve):
    """Two tables that turn pi(t) into a pair of array lookups.

    ``within[k]`` counts the odd primes from the start of k's block through k
    itself; ``prefix[b]`` counts the odd primes in every block before b.
    """
    within = np.cumsum(sieve.reshape(-1, _BLOCK_ODDS), axis=1, dtype=np.uint8)
    prefix = np.zeros(within.shape[0], dtype=np.int64)
    np.cumsum(within[:-1, -1], dtype=np.int64, out=prefix[1:])
    return within.reshape(-1), prefix

def _primes_upto(sieve, cap):
    odd = np.flatnonzero(sieve[: max(cap - 1, 1) // 2 + 1]).astype(np.int64)
    return np.concatenate(([2], 2 * odd + 1))

class GoldbachCounts:
    """Prime counts and Goldbach "dud" counts backed by one cached sieve.

    ``duds_at`` reproduces the nested loops in ``goldbach.py``:

        d(y) = sum of max(0, O(y - p_i * p_j) - j)
               over odd primes p_i <= p_j with p_i * p_j <= y

    where ``O(t) = ceil(t / 2) - pi(t)`` counts the odd non-primes up to t and
    ``j`` is the index of p_j among the primes (index 0 being 2). Every count
    comes from a table lookup, so no argument may exceed ``limit``.

    ``max_prime_index`` truncates the prime list the way running out of loaded
    primes used to; leave it at None for the mathematically complete pair set.
    """

    def __init__(
        self,
        limit,
        max_prime_index=None,
        cache_dir=_SIEVE_CACHE,
        chunk=1 << 20,
    ):
        self.limit = int(limit)
        self.chunk = int(chunk)
        self._max_k = (self.limit - 1) // 2
        sieve = _load_odd_sieve(self.limit, cache_dir)
        # a pair needs p_i >= 3, so no factor above limit // 3 can ever be used
        self.primes = _primes_upto(sieve, max(self.limit // 3, 3))
        if max_prime_index is not None:
            self.primes = self.primes[: int(max_prime_index) + 1]
        self._within, self._prefix = _block_prime_counts(sieve)

    def _check(self, x):
        if x > self.limit:
            raise ValueError(f"{x} is past the sieve limit {self.limit}")
        return x

    def pi(self, t):
        """Number of primes <= t, for a scalar or an array of t."""
        t = np.asarray(t)
        k = np.clip((t - 1) >> 1, 0, self._max_k)
        counts = self._prefix[k >> _BLOCK_SHIFT] + self._within[k] + 1
        return np.where(t < 2, 0, counts)

    def odd_products_upto(self, t):
        """O(t): odd numbers <= t that are not prime."""
        t = np.asarray(t)
        return np.where(t < 1, 0, ((t + 1) >> 1) - self.pi(t))

    def odd_prime_sum_candidates(self, x):
        return max(0, int(self.pi(self._check(2 * (int(x) // 2)))) - 1)

    def relevant_prods_up_to(self, x):
        return max(0, int(self.odd_products_upto(self._check(int(x)) - 3)))

    def duds_at(self, y):
        return self.duds_batch([y])[int(y)]

    def duds_batch(self, ys):
        """d(y) for every y, in one pass over the prime pairs.

        Sharing the pass matters: the pairs are the expensive part and they do
        not depend on y, so a whole table costs barely more than a single row.
        """
        ys = np.unique(np.asarray(list(ys), dtype=np.int64))
        if ys.size == 0:
            return {}
        y_max = self._check(int(ys[-1]))
        totals = np.zeros(ys.size, dtype=np.int64)
        primes = self.primes
        for i in range(1, primes.size):
            p1 = int(primes[i])
            if p1 * p1 > y_max:  # the smallest pair p1 * p1 already overshoots
                break
            hi = int(np.searchsorted(primes, y_max // p1, side="right"))
            for lo in range(i, hi, self.chunk):
                stop = min(lo + self.chunk, hi)
                prods = p1 * primes[lo:stop]
                jidx = np.arange(lo, stop, dtype=np.int64)
                for n in range(ys.size):
                    # clamping at 0 drops the pairs whose product overshoots y,
                    # since O(0) - j is negative for every j >= 1
                    t = np.maximum(ys[n] - prods, 0)
                    term = ((t + 1) >> 1) - self.pi(t) - jidx
                    np.maximum(term, 0, out=term)
                    totals[n] += term.sum()
        return {int(y): int(v) for y, v in zip(ys, totals)}

    def build_table(self, bot, top, step=2):
        """The goldbach.py table for z in range(bot, top, step)."""
        zs = list(range(int(bot), int(top), int(step)))
        duds = self.duds_batch([y for z in zs for y in (z - 1, z)])
        rows = []
        for z in zs:
            new_duds = duds[z] - duds[z - 1]
            relevant = self.relevant_prods_up_to(z)
            rows.append(
                {
                    "z": z,
                    "odd_prime_sum_candidates": self.odd_prime_sum_candidates(z),
                    "duds_to_x": duds[z],
                    "new_duds_for": new_duds,
                    "relevant_prods_up_to": relevant,
                    "primes_prevented": relevant
                    - self.relevant_prods_up_to(z // 2)
                    - new_duds,
                }
            )
        return pd.DataFrame(rows)

@lru_cache(maxsize=2)
def goldbach_counts(limit, max_prime_index=None, cache_dir=_SIEVE_CACHE):
    """Cached GoldbachCounts so repeated calls reuse the sieve and tables."""
    return GoldbachCounts(limit, max_prime_index=max_prime_index, cache_dir=cache_dir)