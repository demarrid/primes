import math
from fractions import Fraction
# https://www.scribd.com/document/829987538/Final-MONZO-code
# Define the prime basis for the monzos (e.g., 2, 3, 5, 7)
PRIMES = [2, 3, 5, 7]


def monzo_to_fraction(monzo):
    """Converts a monzo (prime exponent vector) back to a fraction."""
    numerator = 1
    denominator = 1
    for exponent, prime in zip(monzo, PRIMES):
        if exponent > 0:
            numerator *= prime ** exponent
        elif exponent < 0:
            denominator *= prime ** (-exponent)
    return Fraction(numerator, denominator)


def monzo_multiply(monzo1, monzo2):
    """Multiplies two monzos (adds their prime exponents)."""
    return [m1 + m2 for m1, m2 in zip(monzo1, monzo2)]


def octave_reduce_monzo(monzo):
    """Reduces a monzo to within an octave (1/1 to 2/1) by adjusting the 2-exponent."""
    while monzo_to_fraction(monzo) < 1:  # If the interval is less than 1/1
        monzo[0] += 1  # Multiply by 2 (octave up)
    while monzo_to_fraction(monzo) >= 2:  # If the interval is 2/1 or greater
        monzo[0] -= 1  # Divide by 2 (octave down)
    return monzo


def simplify_monzo(monzo, comma_monzo, max_steps=100, debug=False):
    """Simplifies a monzo by applying comma adjustments iteratively."""
    # Store the reciprocal of the comma monzo
    comma_reciprocal_monzo = [-c for c in comma_monzo]
    simplified_monzo = monzo.copy()

    for step in range(max_steps):
        # Check if applying the comma or its reciprocal simplifies the monzo
        adjusted_monzo_up = [m + c for m, c in zip(simplified_monzo, comma_monzo)]
        adjusted_monzo_down = [m + c for m, c in zip(simplified_monzo, comma_reciprocal_monzo)]

        # Convert monzos to fractions for comparison
        fraction_up = monzo_to_fraction(adjusted_monzo_up)
        fraction_down = monzo_to_fraction(adjusted_monzo_down)
        current_fraction = monzo_to_fraction(simplified_monzo)

        # Compare the simplicity of the adjusted fractions
        if fraction_up.numerator + fraction_up.denominator < current_fraction.numerator + current_fraction.denominator:
            simplified_monzo = adjusted_monzo_up
        elif fraction_down.numerator + fraction_down.denominator < current_fraction.numerator + current_fraction.denominator:
            simplified_monzo = adjusted_monzo_down
        else:
            break  # Stop if no further simplification is possible

        # Apply octave reduction to ensure the interval is within 1/1 to 2/1
        simplified_monzo = octave_reduce_monzo(simplified_monzo)

        if debug:
            print(f"Step {step}: Current: {monzo_to_fraction(simplified_monzo)}")
            print(f"  Adjusted Up: {fraction_up}")
            print(f"  Adjusted Down: {fraction_down}")

    if debug:
        print(f"Final Simplified Interval: {monzo_to_fraction(simplified_monzo)}")

    return simplified_monzo


def normalize_and_adjust(fraction, comma_fraction, max_steps=100, debug=False):
    """Normalizes the fraction to the range [1/1, 2/1] and adjusts it by the comma."""
    # Normalize to the range [1/1, 2/1]
    while fraction >= 2:
        fraction /= 2
    while fraction < 1:
        fraction *= 2

    # Store the reciprocal of the comma
    comma_reciprocal = 1 / comma_fraction

    # Adjust the fraction by the comma iteratively
    for step in range(max_steps):
        # Check if applying the comma or its reciprocal simplifies the fraction
        adjusted_up = fraction * comma_fraction
        adjusted_down = fraction * comma_reciprocal

        # Compare the simplicity of the adjusted fractions
        if adjusted_up.numerator + adjusted_up.denominator < fraction.numerator + fraction.denominator:
            fraction = adjusted_up
        elif adjusted_down.numerator + adjusted_down.denominator < fraction.numerator + fraction.denominator:
            fraction = adjusted_down
        else:
            break  # Stop if no further simplification is possible

        # Normalize to the range [1/1, 2/1] after each adjustment
        while fraction >= 2:
            fraction /= 2
        while fraction < 1:
            fraction *= 2

        if debug:
            print(f"Step {step}: Current: {fraction}")
            print(f"  Adjusted Up: {adjusted_up}")
            print(f"  Adjusted Down: {adjusted_down}")

    if debug:
        print(f"Final Normalized and Adjusted Interval: {fraction}")

    return fraction


def generate_lattice(monzo1, monzo2, comma_monzo, rows, cols, debug=False):
    """Generates a musical lattice using monzos."""
    lattice = []
    for i in range(rows):
        row = []
        for j in range(cols):
            # Scale the monzos by the lattice coordinates
            scaled_monzo1 = [m * (j - cols // 2) for m in monzo1]
            scaled_monzo2 = [m * (i - rows // 2) for m in monzo2]

            # Calculate the interval by stacking the scaled monzos
            interval_monzo = monzo_multiply(scaled_monzo1, scaled_monzo2)
            simplified_monzo = simplify_monzo(interval_monzo, comma_monzo, debug=debug)
            simplified_fraction = monzo_to_fraction(simplified_monzo)

            # Normalize and adjust the fraction
            final_fraction = normalize_and_adjust(simplified_fraction, monzo_to_fraction(comma_monzo), debug=debug)
            row.append(final_fraction)
        lattice.append(row)
    return lattice


def format_lattice(lattice, is_cents=False):
    """Formats the lattice for visually appealing output."""
    if is_cents:
        str_lattice = [[f"{f:.2f}" for f in row] for row in lattice]  # Format cents as strings with 2 decimal places
    else:
        str_lattice = [[str(f) for f in row] for row in lattice]  # Format fractions as strings

    col_widths = [max(len(str_lattice[row][col]) for row in range(len(lattice))) for col in range(len(lattice[0]))]

    formatted_output = ""
    for row in str_lattice:
        for i, cell in enumerate(row):
            formatted_output += cell.rjust(col_widths[i]) + "  "
        formatted_output += "\n"
    return formatted_output


def fraction_to_cents(fraction):
    """Converts a fraction to its logarithmic representation in cents."""
    return 1200 * math.log2(fraction.numerator / fraction.denominator)


# Predefined monzos
monzo1 = [-1, 1, 1, -1]  # 15/14
monzo2 = [-1, -1, 0, 1]  # 7/6
comma_monzo = [-5, 2, 2, -1]  # 225/224

# Example usage
rows = 3
cols = 7
debug = False  # Enable debugging for simplification process

# Generate the lattice in fractions
lattice_fractions = generate_lattice(monzo1, monzo2, comma_monzo, rows, cols, debug=debug)
formatted_lattice_fractions = format_lattice(lattice_fractions)

# Generate the lattice in cents
lattice_cents = [[fraction_to_cents(f) for f in row] for row in lattice_fractions]
formatted_lattice_cents = format_lattice(lattice_cents, is_cents=True)

# Print both lattices
print("Lattice in Fractions:")
print(formatted_lattice_fractions)
print("Lattice in Cents:")
print(formatted_lattice_cents)
