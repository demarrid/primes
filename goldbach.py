from monzo import Monzo, PRIMES

for i in range(4, 30, 2):
    m = Monzo.from_int(i)
    # print(m)
    extended_mods = m.get_extended_modular_coordinates()
    int_val = m.to_int()
    pair = None
    for p1 in PRIMES:
        p2 = int_val - p1
        if p2 in PRIMES:
            pair = (p1, p2)
            break
    z = [k % PRIMES[j] for j, k in enumerate(extended_mods)]
    print(f"{int_val} = {pair[0]} + {pair[1]}:\n{extended_mods} = {Monzo.from_int(pair[0]).get_extended_modular_coordinates()} + {Monzo.from_int(pair[1]).get_extended_modular_coordinates()}")

    #it's 2 4 6 and 1 3 5