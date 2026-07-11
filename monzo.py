from fractions import Fraction
import math

target_max_int = 1_000_0

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
        maxlen = max(len(self.e), len(other.e))
        a_ext = list(self.e) + [0] * (maxlen - len(self.e))
        b_ext = list(other.e) + [0] * (maxlen - len(other.e))
        return Monzo(a + b for a, b in zip(a_ext, b_ext))

    def __sub__(self, other):
        maxlen = max(len(self.e), len(other.e))
        a_ext = list(self.e) + [0] * (maxlen - len(self.e))
        b_ext = list(other.e) + [0] * (maxlen - len(other.e))
        return Monzo(a - b for a, b in zip(a_ext, b_ext))
   

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
        if len(self) == 0:
            self.c = [-1]
        else:
            self.c = [(self.to_int() % PRIMES[i]) - PRIMES[i] for i in range(len(self))]
        return self.c

    def get_extended_modular_coordinates(self):
        mods = self.get_modular_coordinates().copy()
        for i in range(len(mods), len(PRIMES)):
            p = PRIMES[i]
            mods.append((self.to_int() % p) - p)
            if p > self.to_int():
                break
        return mods

    def successor(self):
        return self.succeed(1)

    def succeed(self, k):
        self.to_int()
        modular_coordinates = self.get_modular_coordinates()
        arr = [(1 if mod_coord + (k % PRIMES[i]) == 0 else 0) for i, mod_coord in enumerate(modular_coordinates)]
        val = self.to_int() + k
        if not any(arr):
            new_array = [0] * (len(arr) + 1)
            new_array[-1] = 1
            arr = new_array
            if (len(self) == len(PRIMES)):
                PRIMES.append(val)

        toReturn = self.from_int(val)
        toReturn.self_int = val
        toReturn.c = len(arr) * [0]

        for i in range(len(toReturn.c)):
            if i == len(modular_coordinates):
                toReturn.c[i] = - PRIMES[i]
            else:
                toReturn.c[i] = ((modular_coordinates[i] + k) % PRIMES[i]) - PRIMES[i]
        
        return toReturn

    def next_prime(self):
        modular_coordinates = self.get_modular_coordinates()
        increase = 1
        for candidate in range(1-modular_coordinates[0], PRIMES[len(self)-1], 2):
            if all(modular_coordinates[index] + (candidate % PRIMES[index]) != 0 for index in range(len(modular_coordinates))):
                increase = candidate
                break

        return self.succeed(increase)

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
            cls.get_nth_prime(len(PRIMES) + 1)
            return cls.from_int(n)
        m = cls(exps)
        m.self_int = COPY
        return m

    @classmethod
    def get(cls, n: int):
        return cls.from_int(n)

    @classmethod
    def from_modular_coordinates(cls, coordinates: list[int]):
        for i in range(len(coordinates)):
            if coordinates[i] == 0:
                return cls([-1] * len(coordinates))
            coordinates[i] = (coordinates[i] % PRIMES[i]) - PRIMES[i]

        solved = False
        coefficients = [0] * len(coordinates)
        while not solved:
            target = PRIMES[len(coordinates) - 1]
            for i in range(len(coordinates)):
                int_val = coefficients[i] * PRIMES[i] + (coordinates[i] % PRIMES[i])
                while int_val < target or int_val % PRIMES[i] != (coordinates[i] % PRIMES[i]):
                    coefficients[i] += 1
                    int_val = coefficients[i] * PRIMES[i] + (coordinates[i] % PRIMES[i])
            
            solved = True
            target = max([coefficients[i] * PRIMES[i] + (coordinates[i] % PRIMES[i]) for i in range(len(coordinates))])
            for k in range(len(coordinates)):
                if coefficients[k] * PRIMES[k] + (coordinates[k] % PRIMES[k]) != target:
                    coefficients[k] += 1
                    solved = False
                    break

            
        return cls.from_int(target)


    @classmethod
    def get_nth_prime(cls, n):
        k = n - 1
        while k > len(PRIMES) - 1:
            m = cls.get(PRIMES[:-1])
            m = m.next_prime()
            k -= 1
        return PRIMES[n-1]

    @classmethod
    def get_prime_of_index(cls, index):
        return cls.get_nth_prime(index + 1)

