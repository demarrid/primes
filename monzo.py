from fractions import Fraction
import math
import glob

target_max_int = 100_00_0
max_partition_len = 25

stored_primes = [int(k) for k in open("primes.txt","r").read().replace("\n", ",").split(",")]

if stored_primes[-1] < target_max_int:
    stored_primes = []
    files = sorted(glob.glob("./primelists/100000primes/*"), key=lambda x: x)
    for file in files:
        if len(stored_primes) == 0 or stored_primes[-1] < target_max_int:
            with open(file, "r") as f:
                for line in f:
                    if len(line) > 0:
                        stored_primes.append(int(line))

PRIMES = [int(p) for p in stored_primes if p <= target_max_int]

monzo_cache = {}
color_sequence = ["#83ECF2", "#5FDEB6", "#E3B756", "#3B4AD9", "#22BF4B", "#8117BF", "#BF1773", "#85780B", "#12662F", "#6E0C2F", "#6E0C2F"]

class Monzo:
    __slots__ = ("e", "c", "self_int", "self_norm", "self_square_norm",
             "self_fraction", "self_len", "self_color")

    def __init__(self, exponents, int_val=None):
        self.e = list(exponents)
        self.c = None
        self.self_int = self.self_norm = self.self_square_norm = None
        self.self_fraction = self.self_len = self.self_color = None
        if int_val is not None:
            self.self_int = int_val
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

    def get_coordinates(self):
        if len(self) > max_partition_len *2:
            return self.e[:max_partition_len] + self.e[len(self) - max_partition_len:]
        else:
            return self.e[:self.__len__()]

    def get_modular_coordinates(self):
        if self.c is not None:
            return self.c
        if len(self) == 0:
            self.c = [-1]
        else:
            if len(self) > max_partition_len *2:
                self.c = [(self.to_int() % PRIMES[i]) - PRIMES[i] for i in range(max_partition_len)] + [(self.to_int() % PRIMES[i]) - PRIMES[i] for i in range(len(self) - max_partition_len, len(self))]
            else:
                self.c = [(self.to_int() % PRIMES[i]) - PRIMES[i] for i in range(len(self))]
        return self.c

    def get_positive_modular_coordinates(self):
        return [k % PRIMES[j] for j, k in enumerate(self.get_modular_coordinates())]

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

    def succeed(self, k, to_prime=False):
        self.to_int()
        modular_coordinates = self.get_modular_coordinates()
        arr = [(1 if mod_coord + (k % PRIMES[i]) == 0 else 0) for i, mod_coord in enumerate(modular_coordinates)]
        val = self.to_int() + k
        if not any(arr):
            new_array = [0] * (len(arr) + 1)
            new_array[-1] = 1
            arr = new_array
            if (val > PRIMES[-1]):
                PRIMES.append(val)
        toReturn = self.from_prime_of_index(len(PRIMES) - 1) if to_prime else self.from_int(val)
        toReturn.self_int = val
        toReturn.c = len(arr) * [0]

        for i in range(len(toReturn.c)):
            if i == len(modular_coordinates):
                toReturn.c[i] = - PRIMES[i]
            else:
                toReturn.c[i] = ((modular_coordinates[i] + k) % PRIMES[i]) - PRIMES[i]
        
        return toReturn


    def prime(self):
        return self.norm() == 1

    def next_prime(self):
        if self.to_int() < PRIMES[-1]:
            for i in range(len(self), len(PRIMES)):
                if self.to_int() < PRIMES[i]:
                    return self.from_prime_of_index(i)

        modular_coordinates = self.get_modular_coordinates()
    
        increase = 1
        for candidate in range(1-modular_coordinates[0], PRIMES[len(self)-1], 2):
            if all(modular_coordinates[index] + (candidate % PRIMES[index]) != 0 for index in range(len(modular_coordinates))):
                increase = candidate
                break

        return self.succeed(increase, to_prime=True)

    def to_int(self):
        if self.self_int is not None:
            return self.self_int
        if self.e == [-1] * len(PRIMES):
            self.self_int = 0
            return self.self_int

        self.self_int = math.prod(p ** e for p, e in zip(PRIMES, self.e))
        return self.self_int

    def get_index(self, i):
        return self.e[i] if i < len(self.e) else 0
    
    def with_index(self, i, value):
        new_e = self.e.copy()
        new_e[i] = value
        return Monzo(new_e)

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
        n = int(n)
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
            cls.get_prime_of_index(len(PRIMES))
            return cls.from_int(n)
        m = cls(exps)
        m.self_int = COPY
        return m

    @classmethod
    def from_prime_of_index(cls, i):
        m = cls([0] * (i) + [1], PRIMES[i])
        return m

    @classmethod
    def get(cls, n: int):
        return cls.from_int(n)

    @classmethod
    def from_modular_coordinates(cls, coordinates: list[int]):
        for i in range(len(coordinates)):
            coordinates[i] = (coordinates[i] % PRIMES[i]) - PRIMES[i]
            if coordinates[i] == -PRIMES[i]:
                return cls([-1] * len(coordinates))

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
            m = cls.get(PRIMES[-1])
            m = m.next_prime()
            k -= 1
        return PRIMES[n-1]

    @classmethod
    def get_prime_of_index(cls, index):
        return cls.get_nth_prime(index + 1)

    @classmethod
    def is_prime(cls, n):
        if n in monzo_cache:
            return monzo_cache[n].prime()
        for i in range(len(PRIMES)):
            if PRIMES[i] == n:
                return True
            if PRIMES[i] > n:
                return False
        return False
