from monzo import PRIMES, Monzo, target_max_int
import numpy as np
import random
import os
import pandas as pd
import mpmath as mp

class Complex:
    def __init__(self, real: mp.mpf, imaginary: mp.mpf):
        self.real = real
        self.imaginary = imaginary

    @staticmethod
    def from_polar(r: mp.mpf, theta: mp.mpf):
        return Complex(r * mp.cos(theta), r * mp.sin(theta))
    
    def conjugate(self):
        return Complex(self.real, -self.imaginary)

    def norm(self):
        return self.__abs__()
    
    def __pow__(self, other): 
        return self.to_power(other)

    def angle(self):
        return mp.atan2(self.imaginary, self.real)

    def _coerce(self, other):
        if isinstance(other, Complex):
            return other
        if isinstance(other, complex):
            return Complex(mp.mpf(other.real), mp.mpf(other.imag))
        if isinstance(other, (int, float, mp.mpf)):
            return Complex(mp.mpf(other), mp.mpf(0))
        return NotImplemented

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Complex(self.real + other.real, self.imaginary + other.imaginary)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Complex(self.real - other.real, self.imaginary - other.imaginary)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Complex(self.real * other.real - self.imaginary * other.imaginary, self.real * other.imaginary + self.imaginary * other.real)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Complex((self.real * other.real + self.imaginary * other.imaginary) / (other.real ** 2 + other.imaginary ** 2), (self.imaginary * other.real - self.real * other.imaginary) / (other.real ** 2 + other.imaginary ** 2))
   
    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __repr__(self):
        return f"{self.real} + {self.imaginary}i"
   
    def __eq__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self.real == other.real and self.imaginary == other.imaginary

    def __abs__(self):
        return mp.sqrt(self.real ** 2 + self.imaginary ** 2)
    
    def scalar_mul(self, other: float):
        return Complex(self.real * other, self.imaginary * other)
    
    def to_power(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        # e^(zln|self|)
        pow = other.scalar_mul(mp.log(self.norm()))

        real = mp.e**(pow.real)
        k = Complex(mp.cos(pow.imaginary), mp.sin(pow.imaginary))
        return k.scalar_mul(real)

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

def padic_order(p, fraction: mp.mpf):
    return safe_padic_order(p, fraction.numerator) - safe_padic_order(p, fraction.denominator)

def padic_magnitude(p, fraction: mp.mpf):
    return p ** (-padic_order(p, fraction))

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


# lodone, 2024
def riemann_zeta(s: Complex):
    if s.real <= 1:
        M = 2

        t = s.imaginary
        epsilon = s.real - 0.5
        p = mp.sqrt(t / ( 2 * mp.pi))
        omega = mp.sqrt((2* mp.pi) / t)  # noqa: F841
        N = mp.floor(p)

        sup_err_norm = mp.pow(mp.e, 0.1*t) # for |t| > 100  # noqa: F841

        Z = Complex(0, 0)

        for n in range(1, N+1):
            epsilon_ln_root_no_e = epsilon*mp.log(mp.pow(t/(2*mp.pi *n ** 2), 1/2))
            t_ln_root_minus = t * mp.log(mp.pow(t / (2*mp.e*mp.pi*(n ** 2)), 1/2) - (1/8) * mp.pi)
            reciprocal_root_n = mp.pow(n, -1/2)
            L = mp.cosh(epsilon_ln_root_no_e) * reciprocal_root_n * mp.cos(t_ln_root_minus)
            R = mp.sinh(epsilon_ln_root_no_e) * reciprocal_root_n * mp.sin(t_ln_root_minus)     

            Z += 2 * L
            Z += Complex(0, 2) * R

        R = 0

        for j in range(M + 1):
            C_j_p_epsilon = Complex(0, 0)
            if j == 0:
                C_j_p_epsilon = mp.cos(2 * mp.pi * (mp.pow(p, 2) - p - (1/16))) / mp.cos(2 * mp. pi * p) # edwards, riemann's zeta function, 154
            if j == 1:
                # involves 3rd order derivatives
                C_j_p_epsilon = Complex(0, 0)
            if j == 2:
                # involves 6th order derivatives
                C_j_p_epsilon = Complex(0, 0) 
            else:
                raise ValueError(f"Invalid j: {j}")

            R += C_j_p_epsilon * (2 * mp.pi / t) ** (j / 2)
        
        R *= (-1) ** (N - 1) * (2 * mp.pi / t) ** (1/4)

        Z += R

        # our error interval is [Z - sup_err_norm, Z + sup_err_norm]

        return Z

    return euler_product(s)

def euler_product(s: Complex):
    r = Complex(1, 0)
    negative_s = Complex(0, 0) - s
    max_p = PRIMES[-1]
    p = 2
    k = 1
    while p < target_max_int:
        r *= Complex(1, 0) / (Complex(1, 0) - Complex(p, 0) ** negative_s)
        k += 1
        p = Monzo.from_prime_of_index(k - 2).next_prime().to_int() if p >= max_p else PRIMES[k-1]
        if k % 100000 == 0:
            print(f"Performing multiplication of {k}/{len(PRIMES) + 1}th prime")
    return r

def to_riemann_harmonic(s: Complex):
    return lambda x: (x**s.real / (mp.ln(x) * s.norm())) * mp.cos(s.imaginary * mp.ln(x) - mp.atan(s.imaginary / s.real))

def von_mangoldt_function(n: int):
    m = Monzo.get(n)
    k = -1
    for i in range(len(m)):
        if m.get_index(i) > 0:
              if k == -1:
                k = i
              else:
                return 0
    return mp.log(PRIMES[k])

def willans(n: int):
    k = 1

    for i in range(1, 2** n + 1):
        acc = 0
        for j in range(1, i+1):
            acc += mp.floor ((mp.cos(mp.pi * (mp.factorial(j-1)+1)/ j) ** 2))
        k += mp.floor((n / acc) ** (1/n))

    return k

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
    return 1 / (1 + mp.exp(-x))