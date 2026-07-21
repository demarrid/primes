from monzo import Monzo

for i, p in enumerate([Monzo.get_prime_of_index(i) for i in range(8)]):
    for m in range(1, p -1):
        arr_1 = [-1] * (i)
        arr_1.append(-m)
        arr_2 = [1] * (i)
        arr_2.append(-m + 2)
        d = Monzo.from_modular_coordinates(arr_1)
        e = Monzo.from_modular_coordinates(arr_2)
        print(f"[{d.to_int()}, {e.to_int()}]") # has float in 2nd pos when p_n and p_{n+2} are twins

        # can expand by additionally adding 2 on internal primes