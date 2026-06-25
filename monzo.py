from fractions import Fraction
import math

dim_count = 2000

primes = open("primes.txt","r").read().replace("\n", ",").split(",")
PRIMES = [int(p) for p in primes[:dim_count]]

color_sequence = ["#83ECF2", "#5FDEB6", "#E3B756", "#3B4AD9", "#22BF4B", "#8117BF", "#BF1773", "#85780B", "#12662F", "#6E0C2F", "#6E0C2F"]

class Monzo:
    __slots__ = ("e",)

    def __init__(self, exponents):
        self.e = list(exponents)

    def __add__(self, other):
        return Monzo(a + b for a, b in zip(self.e, other.e))

    def __sub__(self, other):
        return Monzo(a - b for a, b in zip(self.e, other.e))

    def __mul__(self, k): 
        return Monzo(k * a for a in self.e)

    def __truediv__(self, k):
        return Monzo(a / k for a in self.e)

    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    __floordiv__ = __truediv__
    __rfloordiv__ = __rtruediv__

    def __matmul__(self, val): 
        return sum(a * b for a, b in zip(self.e, val))

    def square_norm(self):
        return sum(a ** 2 for a in self.e)

    def norm(self):
        return math.sqrt(self.square_norm())

    def __eq__(self, other):
        return isinstance(other, Monzo) and self.e == other.e

    def __repr__(self):
        return f"Monzo({self.e})"

    def to_fraction(self):
        f = Fraction(1)
        for exp, p in zip(self.e, PRIMES):
            f *= Fraction(p) ** exp
        return f

    def to_int(self):
        if self.e == [-1] * len(PRIMES):
            return 0
        return math.prod(p ** e for p, e in zip(PRIMES, self.e))

    def get_index(self, i):
        return self.e[i]

    def as_color(self):
        if self.e == [-1] * len(PRIMES):
            return "#000000"
        return color_sequence[(self.to_int() - 1) % len(color_sequence)]

    def __len__(self):
        return len(self.e)

    @classmethod
    def from_int(cls, n):
        if n < 1:
            return cls([-1] * len(PRIMES))
        exps = [0] * len(PRIMES)
        for i, p in enumerate(PRIMES):
            while n % p == 0:
                exps[i] += 1
                n //= p
        if n != 1:
            raise ValueError(f"{n} has a prime factor outside PRIMES")
        return cls(exps)

    @classmethod
    def get_nth_prime(cls, n):
        return PRIMES[n-1]

integer_to_monzo = {}

def get_monzo(integer):
    if integer not in integer_to_monzo:
        integer_to_monzo[integer] = Monzo.from_int(integer)
    return integer_to_monzo[integer]
