from fractions import Fraction
from monzo import PRIMES, Monzo
import numpy as np
import random
import os
import pandas as pd

class Complex:
    def __init__(self, real: Fraction, imaginary: Fraction):
        self.real = real
        self.imaginary = imaginary

    @staticmethod
    def from_polar(r: Fraction, theta: Fraction):
        return Complex(r * np.cos(theta), r * np.sin(theta))
    
    def conjugate(self):
        return Complex(self.real, -self.imaginary)

    def norm(self):
        return self.__abs__()
    
    def __pow__(self, n: int):
        return Complex(self.real ** n, self.imaginary ** n)

    def angle(self):
        return np.arctan2(self.imaginary, self.real)

    def __add__(self, other):
        return Complex(self.real + other.real, self.imaginary + other.imaginary)

    def __sub__(self, other):
        return Complex(self.real - other.real, self.imaginary - other.imaginary)

    def __mul__(self, other):
        return Complex(self.real * other.real - self.imaginary * other.imaginary, self.real * other.imaginary + self.imaginary * other.real)

    def __truediv__(self, other):
        return Complex((self.real * other.real + self.imaginary * other.imaginary) / (other.real ** 2 + other.imaginary ** 2), (self.imaginary * other.real - self.real * other.imaginary) / (other.real ** 2 + other.imaginary ** 2))
   
    def __repr__(self):
        return f"{self.real} + {self.imaginary}i"
   
    def __eq__(self, other):
        return self.real == other.real and self.imaginary == other.imaginary

    def __abs__(self):
        return np.sqrt(self.real ** 2 + self.imaginary ** 2)
    

def hsv_to_hex(h, s, v):
    # Normalize S and V to 0-1
    s /= 100.0
    v /= 100.0

    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c

    r1, g1, b1 = 0.0, 0.0, 0.0

    if 0 <= h < 60:
        r1, g1, b1 = c, x, 0
    elif 60 <= h < 120:
        r1, g1, b1 = x, c, 0
    elif 120 <= h < 180:
        r1, g1, b1 = 0, c, x
    elif 180 <= h < 240:
        r1, g1, b1 = 0, x, c
    elif 240 <= h < 300:
        r1, g1, b1 = x, 0, c
    elif 300 <= h < 360:
        r1, g1, b1 = c, 0, x

    # Convert to 0-255 range
    r = round((r1 + m) * 255)
    g = round((g1 + m) * 255)
    b = round((b1 + m) * 255)

    # Format as 2-digit uppercase hexadecimal
    res = f"#{r:02X}{g:02X}{b:02X}"

    return res

def safe_padic_order(p, n: int):
    if n == 1:
        return 0

    monzo = Monzo.get(n)
    indx = PRIMES.index(p)
    return monzo.get_index(indx) if indx > -1 else -1

def padic_order(p, fraction: Fraction):
    return safe_padic_order(p, fraction.numerator) - safe_padic_order(p, fraction.denominator)

def padic_magnitude(p, fraction: Fraction):
    return p ** (-padic_order(p, fraction))

# \sum_{n=1}^{\infty} \frac{\mu(n)}{n^s} = \frac{1}{\zeta(s)}

def mobius_function(n: int):
    if n == 1:
        return 1
    monzo = Monzo.get(n)
    if any(monzo.get_index(i) > 1 for i in range(len(monzo))):
        return 0
    return liouville_function(n)

def liouville_function(n: int):
    return (-1) ** Monzo.get(n).square_norm()

def harmonic_number(n: int):
    return sum(1 / i for i in range(1, n + 1))

def euler_totient_function(n: int):
    return sum(1 for i in range(1, n) if Monzo.get(i) * Monzo.get(n) == 0)

def cyclotomic_field(n: int):
    roots_of_unity = [ Complex(1, 2 * np.pi * i / n) for i in range(n)] 

    # we would return cartesian prod of this and q
    
    return roots_of_unity

# algebraic number: root of a monic polynomial with integer coefficients

# elliptic curve: y^2 = x^3 + ax + b


def generate_random_elliptic_curve(n: int):
    x_0 = random.randint() % n
    y_0 = random.randint() % n
    a = random.randint() % n
    b = (y_0 ** 2 - x_0 ** 3 - a * x_0) % n

    return (x_0,y_0,a,b)

def find_factor(n: int):
    ec = generate_random_elliptic_curve(n)
    # add until we find a factor
    P = (ec[0], ec[1])
    m = lambda point : 3*(point[0] ** 2) - ec[3] / (2 * point[1])  # noqa: E731

    m_prime = lambda p0, k : [(k ** 2) - (2 * p0[0]), k * (p0[0] - k) - p0[1] ]  # noqa: E731

    _1p= m(P)
    _2p= m_prime(P, _1p)

    # if gcd(_2p[0], n) \ne 1, then we have found a factor

    # let u, v, so gcd(u, v) \ne 1, and slope between P and Q on curve = u/v

    # compute [m]P mod n

def modular_curve_to_elliptic_curve(n: int):
    # complex upper half plane mod congurence subgroup gamma of modular group SL(2, Z)
    print("hi")

def get_dual_prime_group(prime_index: int):
    p = PRIMES[prime_index]
    return [lambda a: (a * i) % p for i in range(p)]

def to_dirichlet_character(q: int, character: lambda a: int):
    return lambda n: (character(n) if Monzo.get(n) * Monzo.get(q) == 0 else 0)

def riemann_zeta(s: Complex):
    r = 1

    for i in range(len(PRIMES)):
        p = PRIMES[i]
        r *= 1 / (1 - p ** (-s))
        
    return r

def von_mangoldt_function(n: int):
    m = Monzo.get(n)
    k = -1
    for i in range(len(m)):
        if m.get_index(i) > 0:
              if k == -1:
                k = i
              else:
                return 0
    return np.log(PRIMES[k])

# beware of clanker code below

def load_or_build(path, build_fn, keep=True):
    if os.path.exists(path):
        return pd.read_csv(path)
    df = build_fn()
    df.to_csv(path, index=False)
    return df if keep else None   # discard from memory when not needed

def make_spf(N):
    spf = np.arange(N + 1)
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            spf[i*i::i] = np.minimum(spf[i*i::i], i)
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

def sigmoid(x):
    x*=3
    return 1 / (1 + np.exp(-x))