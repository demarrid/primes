import mpmath

from monzo import Monzo

mpmath.mp.dps = 1000

sum = 0
prod = 1


for i in range(100):
    prod *= Monzo.get_prime_of_index(i)
    sum += 1/mpmath.mpf(prod)

print(sum)