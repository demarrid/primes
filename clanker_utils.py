import os
import importlib
import math
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
    """Smallest prime factor of every n <= N.

    Costs 4 bytes per entry while N fits in uint32 (8 past that), so this is
    only the right tool for factoring many numbers below N; for a few large
    ones use ``factor_big``.
    """
    dtype = np.uint32 if N <= np.iinfo(np.uint32).max else np.int64
    spf = np.arange(N + 1, dtype=dtype)
    for i in range(2, math.isqrt(N) + 1):
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

_TRIAL_LIMIT = 1000
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_MR_DETERMINISTIC = 3317044064679887385961981  # _MR_BASES are exact below this
_BRENT_BATCH = 128  # steps between gcds in Pollard-Brent
_BRENT_MAX_R = 1 << 16  # ~0.3 s of rho, enough for factors up to ~1e10; then ECM
_BRENT_TRIES = 3  # polynomials to try before giving up on rho

@lru_cache(maxsize=1)
def _trial_primes():
    return [int(p) for p in primes_upto(_TRIAL_LIMIT, cache_dir=None)]

def is_prime(n):
    """Miller-Rabin, deterministic below 3.3e24; sympy's BPSW above that."""
    n = int(n)
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    if n >= _MR_DETERMINISTIC:
        from sympy import isprime
        return isprime(n)
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _pollard_brent(n, max_r=_BRENT_MAX_R, tries=_BRENT_TRIES):
    """A nontrivial factor of an odd composite n, or None if rho runs out.

    Rho needs about sqrt(p) steps to find a factor p, so it stalls once every
    factor is large; the caller is expected to fall back to ECM then.
    """
    for c in range(1, min(n, tries + 1)):
        y, r, q, g = 2, 1, 1, 1
        x = ys = y
        while g == 1:
            if r > max_r:
                return None
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(_BRENT_BATCH, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += _BRENT_BATCH
            r *= 2
        if g == n:
            # the batched product overshot; replay the last batch one step at a time
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g
    return None

def factor_big(m, spf=None):
    """yield (prime, exponent) for any integer m >= 1, smallest prime first.

    Same output as ``factor`` but needs no sieve: trial division by the primes
    below 1000, then a short Pollard-Brent run on whatever is left. Pieces
    that rho cannot crack go to sympy's factorint, whose ECM handles factors
    of 20+ digits (seconds at that size, growing fast past it). When ``spf``
    is given and m fits in it, the sieve is used instead.
    """
    m = int(m)
    if spf is not None and m < len(spf):
        yield from factor(m, spf)
        return
    counts = {}
    for p in _trial_primes():
        if p * p > m:
            break
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            counts[p] = e
    stack = [m] if m > 1 else []
    while stack:
        n = stack.pop()
        if is_prime(n):
            counts[n] = counts.get(n, 0) + 1
            continue
        d = _pollard_brent(n)
        if d is None:
            from sympy import factorint
            for p, e in factorint(n).items():
                counts[p] = counts.get(p, 0) + e
            continue
        stack.extend((d, n // d))
    for p in sorted(counts):
        yield p, counts[p]

def format_factorization(pairs):
    """``'2^3 * 3 * 5^2'`` from (prime, exponent) pairs; ``'1'`` when empty."""
    return " * ".join(str(p) if e == 1 else f"{p}^{e}" for p, e in pairs if e) or "1"

def factorization_string(n, spf=None):
    """Prime factorization of an integer n >= 1 as a string, e.g. 360 -> '2^3 * 3^2 * 5'."""
    n = int(n)
    if n < 1:
        raise ValueError("only positive integers have a prime factorization")
    return format_factorization(factor_big(n, spf))

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

_SIEVE_GRANULARITY = 1 << 20
_MAX_CACHED_LIMIT = 1 << 27  # a 64 MB cache file; past this, sieve in memory
_MAX_SIEVE_BYTES = 32 << 30  # only to catch the hopeless ask, not to police big ones

def primes_upto(limit, cache_dir=_SIEVE_CACHE, max_bytes=_MAX_SIEVE_BYTES):
    """Every prime <= ``limit`` as an int64 array.

    Sieve limits are rounded up so that nearby requests share one cache file;
    a sieve built past ``limit`` still answers correctly below it. Very large
    sieves skip the cache rather than leave hundreds of megabytes on disk, and
    a hopeless one is refused up front instead of asking numpy for more memory
    than the machine has.
    """
    limit = int(limit)
    if limit < 2:
        return np.empty(0, dtype=np.int64)
    padded = -(-limit // _SIEVE_GRANULARITY) * _SIEVE_GRANULARITY
    needed = padded // 2 + 1
    if needed > max_bytes:
        raise MemoryError(
            "sieving to {:.3g} needs {:.1f} GiB for the flags alone; raise "
            "max_bytes if the machine really has it".format(limit, needed / (1 << 30))
        )
    if padded > _MAX_CACHED_LIMIT:
        cache_dir = None
    return _primes_upto(_load_odd_sieve(padded, cache_dir), limit)

def first_primes(count, cache_dir=_SIEVE_CACHE):
    """The first ``count`` primes, as Python ints."""
    count = int(count)
    if count <= 0:
        return []
    # p_k < k * (ln k + ln ln k) for k >= 6; the loop covers the small cases
    limit = max(15, int(count * (math.log(count) + math.log(math.log(count) + 1))))
    while True:
        primes = primes_upto(limit, cache_dir)
        if primes.size >= count:
            return [int(p) for p in primes[:count]]
        limit *= 2

def primorials(count, cache_dir=_SIEVE_CACHE):
    """``primorials(k)[i]`` is the product of the first ``i + 1`` primes."""
    out = []
    prod = 1
    for p in first_primes(count, cache_dir):
        prod *= p
        out.append(prod)
    return out

def primes_between(a, b, cache_dir=_SIEVE_CACHE):
    """Primes p with ``a <= p <= b``."""
    if b < 2 or b < a:
        return np.empty(0, dtype=np.int64)
    primes = primes_upto(b, cache_dir)
    return primes[int(np.searchsorted(primes, a, side="left")) :]

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

_PI_MAX_DPS = 1000  # a value this stubborn is a near-integer nobody will meet
_PI_SPARE_DIGITS = 30  # fractional digits to keep once the integer part is paid for

def pi_floor(x):
    """floor(x / ln x), the prime-count estimate, computed exactly.

    The starting precision covers the integer part of the quotient plus 30
    fractional digits, so the very first evaluation resolves the floor for any
    ordinary value. Precision then climbs only for values that sit within
    10**-20 of an integer, and the tolerance test, not the digit count, is
    what allows an answer out.
    """
    import mpmath

    x = float(x)
    dps = _PI_SPARE_DIGITS + max(0, int(math.log10(x)) if x > 1 else 0)
    while True:
        with mpmath.workdps(dps):
            value = mpmath.mpf(x)
            quotient = value / mpmath.ln(value)
            whole = mpmath.floor(quotient)
            frac = quotient - whole
            tol = abs(quotient) * mpmath.mpf(10) ** (10 - dps)
            if tol < frac < 1 - tol or dps >= _PI_MAX_DPS:
                return int(whole)
        dps = min(dps * 4, _PI_MAX_DPS)

def pi_floor_batch(xs, guard_rel=1e-13, guard_abs=1e-9):
    """floor(x / ln x) for many x at once, still exact.

    float64 settles almost every value: x itself is exact and log then divide
    cost a few ulps, so the floor is only in doubt when the quotient sits
    right next to an integer. Those go back through ``pi_floor``. The default
    band is a hundred times a generous bound on that float error, which still
    leaves only a handful of values per hundred thousand on the slow path.

    Since the band is relative, it widens past 1 once the quotient nears
    2**53 and the fraction stops meaning anything; those values all go to
    mpmath, which is the only way to floor them anyway.
    """
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(xs <= 1.0):
        raise ValueError("x / ln(x) is only meaningful for x > 1")

    quotient = xs / np.log(xs)
    if quotient.max() >= 2.0**62:
        raise OverflowError(
            "x / ln(x) reaches {:.3g}, past what an int64 count can hold; "
            "this z is too large for the array path".format(quotient.max())
        )
    whole = np.floor(quotient)
    frac = quotient - whole
    guard = guard_rel * quotient + guard_abs
    unsure = (frac < guard) | (frac > 1.0 - guard)

    out = np.zeros(xs.shape, dtype=np.int64)
    settled = ~unsure
    out[settled] = whole[settled].astype(np.int64)
    for i in np.flatnonzero(unsure):
        out[i] = pi_floor(float(xs[i]))
    return out

_SUM_CHUNK = 1 << 26  # both half-sums then stay four orders below the int64 roof

def exact_sum(counts):
    """Add up a non-negative int64 array without int64 wrapping the total.

    Each element fits, but their sum need not: past s = 16 the twins total
    runs to 10**20. Splitting every value into two 32 bit halves keeps both
    running totals small enough to add in int64, and the halves are recombined
    as Python ints at the end.
    """
    counts = np.asarray(counts)
    total = 0
    for start in range(0, counts.size, _SUM_CHUNK):
        block = counts[start : start + _SUM_CHUNK]
        total += (int((block >> 32).sum()) << 32) + int((block & 0xFFFFFFFF).sum())
    return total

def _iroot(a, k):
    """floor(a ** (1 / k)) for a >= 0, by integer Newton so nothing rounds."""
    if k == 1 or a < 2:
        return a
    r = 1 << -(-a.bit_length() // k)
    while True:
        step = ((k - 1) * r + a // r ** (k - 1)) // k
        if step >= r:
            return r
        r = step

def cofactor_limit(z, n):
    """The largest c worth considering: ``z ** ((n - 1) / n)``, floored.

    A c above this leaves less than ``z ** (1 / n)`` for the final prime, so
    it cannot be the small end of an n-almost-prime. At n == 2 this is the
    ``sqrt(z)`` cutoff the original code used.
    """
    return _iroot(int(z) ** (n - 1), n)

def _cofactor_blocks(limit, k, min_index, primes, prefix=1):
    """Blocks of the k-almost-primes <= limit drawn from ``primes[min_index:]``.

    Factors are picked in non-decreasing index order, so each product turns up
    exactly once however many ways it could be assembled. A block shares one
    prefix and varies only the last factor, which keeps that innermost step a
    single array multiply.
    """
    if k == 1:
        hi = int(np.searchsorted(primes, limit // prefix, side="right"))
        if hi > min_index:
            yield prefix * primes[min_index:hi]
        return
    for j in range(min_index, primes.size):
        p = int(primes[j])
        if prefix * p**k > limit:
            break
        yield from _cofactor_blocks(limit, k - 1, j, primes, prefix * p)

def cofactor_blocks(z, s, n=2, cache_dir=_SIEVE_CACHE):
    """The c values twins.py sums over, a block at a time.

    c runs over the distinct (n-1)-almost-primes up to ``cofactor_limit`` whose
    prime factors are all at least p_s. For n == 2 that is exactly the primes
    from p_s to sqrt(z) that the original built from the loaded prime list.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    limit = cofactor_limit(z, n)
    primes = primes_upto(limit, cache_dir)
    return _cofactor_blocks(limit, n - 1, max(int(s) - 1, 0), primes)

def cofactor_candidates(z, s, n=2, cache_dir=_SIEVE_CACHE):
    """Every c value in one array; see ``cofactor_blocks`` for what they are.

    Past n == 2 there can be hundreds of millions of them, so prefer
    ``twins_row``, which never holds more than one block.
    """
    blocks = [b for b in cofactor_blocks(z, s, n, cache_dir) if b.size]
    if not blocks:
        return np.empty(0, dtype=np.int64)
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

_EXACT_DOUBLE = 1 << 53

def _quotients(z, cs):
    """``z / c`` as float64, matching Python's int division bit for bit."""
    cs = np.asarray(cs, dtype=np.int64)
    if int(z) <= _EXACT_DOUBLE and cs.size and int(cs.max()) <= _EXACT_DOUBLE:
        # both sides are exact doubles here, so one IEEE divide rounds the true
        # quotient exactly the way int.__truediv__ would
        return np.float64(z) / cs.astype(np.float64)
    return np.fromiter((z / int(c) for c in cs), dtype=np.float64, count=cs.size)

def twins_row(z, s, n=2, cs=None, cache_dir=_SIEVE_CACHE):
    """``(how many c values, actual_count)`` for one row of twins.py.

    actual_count is the sum of max(0, pi(z / c) - s) over those c. The c values
    are consumed one block at a time, so a row with a hundred million
    cofactors still only needs memory for the sieve.
    """
    blocks = [np.asarray(cs)] if cs is not None else cofactor_blocks(z, s, n, cache_dir)
    size = 0
    total = 0
    for block in blocks:
        if block.size == 0:
            continue
        size += int(block.size)
        counts = pi_floor_batch(_quotients(z, block))
        counts -= int(s)
        np.maximum(counts, 0, out=counts)
        total += exact_sum(counts)
    return size, total

def twins_actual_count(z, s, n=2, cs=None, cache_dir=_SIEVE_CACHE):
    """twins.py's ``actual_count`` for one s: sum of max(0, pi(z / c) - s)."""
    return twins_row(z, s, n, cs, cache_dir)[1]