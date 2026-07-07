from fractions import Fraction
import math

target_max_int = 1_00

stored_primes = [int(k) for k in open("primes.txt","r").read().replace("\n", ",").split(",")]
PRIMES = [int(p) for p in stored_primes if p <= target_max_int]

monzo_cache = {}

color_sequence = ["#83ECF2", "#5FDEB6", "#E3B756", "#3B4AD9", "#22BF4B", "#8117BF", "#BF1773", "#85780B", "#12662F", "#6E0C2F", "#6E0C2F"]

class Monzo:
    __slots__ = ("e", "c", "self_int", "self_norm", "self_square_norm",
             "self_fraction", "self_len", "self_color")

    def __init__(self, exponents):
        self.e = list(exponents)
        self.c = None
        self.self_int = self.self_norm = self.self_square_norm = None
        self.self_fraction = self.self_len = self.self_color = None
        monzo_cache[self.to_int()] = self

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
        if self.self_square_norm is not None:
            return self.self_square_norm
        self.self_square_norm = sum(a ** 2 for a in self.e)
        return self.self_square_norm

    def norm(self):
        if self.self_norm is not None:
            return self.self_norm
        self.self_norm = math.sqrt(self.square_norm())
        return self.self_norm

    def __eq__(self, other):
        return isinstance(other, Monzo) and self.e == other.e

    def __repr__(self):
        return f"Monzo({self.e[:self.__len__()]})"

    def to_fraction(self):
        if self.self_fraction is not None:
            return self.self_fraction
        f = Fraction(1)
        for exp, p in zip(self.e, PRIMES):
            f *= Fraction(p) ** exp
        self.self_fraction = f
        return self.self_fraction

    def get_modular_coordinates(self):
        if self.c is not None:
            return self.c
        self.c = [self.to_int() % PRIMES[i]  for i in range(len(self))]
        return self.c

    def successor(self):
        return self.succeed(1)

    def succeed(self, k):
        self.to_int()
        arr = [(1 if mod_coord + k == PRIMES[i] else 0) for i, mod_coord in enumerate(self.get_modular_coordinates())]
        if not any(arr):
            arr = arr + [1]
        return Monzo(arr)

    def to_int(self):
        if self.self_int is not None:
            return self.self_int
        if self.e == [-1] * len(PRIMES):
            self.self_int = 0
            return self.self_int

        self.self_int = math.prod(p ** e for p, e in zip(PRIMES, self.e))
        self.get_modular_coordinates()
        return self.self_int

    def get_index(self, i):
        return self.e[i] if i < len(self.e) else 0

    def as_color(self):
        if self.self_color is not None:
            return self.self_color
        if self.e == [-1] * len(PRIMES):
            self.self_color = "#000000"
            return self.self_color

        self.self_color = color_sequence[(self.to_int() - 1) % len(color_sequence)]
        return self.self_color

    def __len__(self):
        if self.self_len is not None:
            return self.self_len

        for i in range(len(self.e))[::-1]:
            if self.e[i] > 0:
                self.self_len = i + 1
                return self.self_len

        self.self_len = 0
        return self.self_len

    @classmethod
    def from_int(cls, n):
        COPY = n
        if n < 1:
            return cls([-1] * len(PRIMES))
        if n in monzo_cache:
            return monzo_cache[n]
        exps = [0] * len(PRIMES)
        for i, p in enumerate(PRIMES):
            while n % p == 0:
                exps[i] += 1
                n //= p
        if n != 1:
            raise ValueError(f"{n} has a prime factor outside PRIMES")
        m = cls(exps)
        m.self_int = COPY
        return m

    @classmethod
    def get(cls, n: int):
        return cls.from_int(n)

    @classmethod
    def get_nth_prime(cls, n):
        return PRIMES[n-1]

