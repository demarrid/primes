import time
import mpmath as mp
from utils import riemann_zeta, some_riemann_zeros
mp.dps = 100

start_time = time.time()
zeros = some_riemann_zeros()
z = zeros[-1]

zeta, sup_err_norm = riemann_zeta(z)
print(zeta)
print(f"Error interval: [{zeta - sup_err_norm}, {zeta + sup_err_norm}]")
if z.real > 1:
    print(f"Time taken: {time.time() - start_time} seconds")