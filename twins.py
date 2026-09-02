import mpmath
import pandas as pd

from clanker_utils import cofactor_limit, first_primes, primorials, twins_row

mpmath.mp.dps = 50

MAX_SIEVE_BYTES = 4 << 30

sum = 0

PRIMES = first_primes(30)
primorial_map = primorials(30)

for s in range(30):
    sum += 1 / mpmath.mpf(primorial_map[s])

rho = sum


def fmt(val):
    return f"{float(val):.5f}"


for n in range(2, 6):
    df = pd.DataFrame(columns=["s", "p_s", "z", "|V|", f"C{n}", "ratio"])

    for s in range(5, 16):
        z = primorial_map[s - 1]
        p_s = PRIMES[s - 1]

        sieve_bytes = cofactor_limit(z, n) // 2
        if sieve_bytes > MAX_SIEVE_BYTES:
            print(f"n={n} s={s}: would need a {sieve_bytes / (1 << 30):.1f} GiB sieve, stopping")
            break

        n_prods, actual_count = twins_row(z, s, n)

        V = 0.56 * z / mpmath.ln(mpmath.ln(z))
        ratio = V / actual_count if actual_count else mpmath.nan

        print("--------------------------------")
        print("n:", n)
        print("s:", s)
        print("p_s:", p_s)
        print("z:", z)
        print("cofactors:", n_prods)
        print("|V|:", (int)(V))
        print(f"C{n}:", actual_count)
        print("ratio:", fmt(ratio))
        print("--------------------------------")

        df = df._append({
            "s": s,
            "p_s": p_s,
            "z": z,
            "|V|": fmt(V),
            f"C{n}": actual_count,
            "ratio": fmt(ratio),
        }, ignore_index=True)

    df.to_csv(f"twins_{n}.csv", index=False)