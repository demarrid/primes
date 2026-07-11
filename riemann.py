from monzo import Monzo, PRIMES
from utils import Complex, riemann_zeta
import math
from sympy import log
z = Complex(1, 1)



# (e^(log |z|))^{z}

# z= a + bi

# e^(0.5ln (a^2+b^2) * (a+bi))

a = 0.5
b = 1

for p in PRIMES[:10]:
    print(math.e**(log(p) * (z.real))*(math.cos(z.imaginary*log(p))+1j*math.sin(z.imaginary*log(p))))