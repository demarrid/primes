from monzo import Monzo

for m in range(1, 7):
    d = Monzo.from_modular_coordinates([-1, -1, -1, -m])
    print(d)
    print(d.get_modular_coordinates())
    print(d.to_int())
    e = Monzo.from_modular_coordinates([-1, -2, 1, -m + 2])
    print(e)
    print(e.get_modular_coordinates())
    print(e.to_int())