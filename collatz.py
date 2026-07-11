from monzo import Monzo, PRIMES

funny = 50
m = Monzo.from_int(funny)

while True:
    # print(m)
    m = m - Monzo([m.get_index(0)])
    if (m.to_int() == 1):
        break
    m = (m + Monzo([0,1])).successor()    
    print(m.to_int())
    print([k % PRIMES[j] for j, k in enumerate(m.get_modular_coordinates())])

    # 3x + 1 is always 1 mod the prime space it's in lololol