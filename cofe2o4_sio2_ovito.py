import numpy as np

# ==================================================
# CoFe2O4@SiO2 core-shell model for OVITO
# Output: cofe2o4_sio2.xyz
# ==================================================

np.random.seed(10)

output_file = "cofe2o4_sio2.xyz"

# -----------------------------
# Core parameters: CoFe2O4
# -----------------------------
a = 8.39
n_cells = 4

core_radius = 14.0      # Angstrom
shell_thickness = 5.0   # Angstrom
outer_radius = core_radius + shell_thickness

center = np.array([
    n_cells * a / 2,
    n_cells * a / 2,
    n_cells * a / 2
])

# -----------------------------
# CoFe2O4 spinel positions
# -----------------------------
tetra_Fe = [
    (1/8, 1/8, 1/8),
    (1/8, 5/8, 5/8),
    (5/8, 1/8, 5/8),
    (5/8, 5/8, 1/8),
    (7/8, 7/8, 7/8),
    (7/8, 3/8, 3/8),
    (3/8, 7/8, 3/8),
    (3/8, 3/8, 7/8),
]

octa_sites = [
    (1/2, 1/2, 1/2),
    (1/2, 1/4, 1/4),
    (1/4, 1/2, 1/4),
    (1/4, 1/4, 1/2),
    (1/2, 3/4, 3/4),
    (3/4, 1/2, 3/4),
    (3/4, 3/4, 1/2),
    (3/4, 1/4, 1/4),
    (1/4, 3/4, 1/4),
    (1/4, 1/4, 3/4),
    (3/4, 3/4, 1/4),
    (3/4, 1/4, 3/4),
    (1/4, 3/4, 3/4),
    (0, 0, 0),
    (0, 1/2, 1/2),
    (1/2, 0, 1/2),
]

u = 0.261
oxygen_base = [
    (u, u, u),
    (u, 0.5-u, 0.5+u),
    (0.5-u, 0.5+u, u),
    (0.5+u, u, 0.5-u),
]

fcc_translations = [
    (0, 0, 0),
    (0, 0.5, 0.5),
    (0.5, 0, 0.5),
    (0.5, 0.5, 0),
]

atoms = []

# -----------------------------
# Generate CoFe2O4 spherical core
# -----------------------------
for i in range(n_cells):
    for j in range(n_cells):
        for k in range(n_cells):
            cell_shift = np.array([i, j, k], dtype=float)

            for pos in tetra_Fe:
                r = (np.array(pos) + cell_shift) * a
                if np.linalg.norm(r - center) <= core_radius:
                    atoms.append(("Fe", r))

            for idx, pos in enumerate(octa_sites):
                r = (np.array(pos) + cell_shift) * a
                if np.linalg.norm(r - center) <= core_radius:
                    element = "Co" if idx < 8 else "Fe"
                    atoms.append((element, r))

            oxygen_positions = []
            for base in oxygen_base:
                for trans in fcc_translations:
                    p = (np.array(base) + np.array(trans)) % 1.0
                    oxygen_positions.append(tuple(p))

            oxygen_positions += [(1-x, 1-y, 1-z) for x, y, z in oxygen_positions]

            for pos in oxygen_positions:
                r = (np.array(pos) + cell_shift) * a
                if np.linalg.norm(r - center) <= core_radius:
                    atoms.append(("O", r))

# -----------------------------
# Generate amorphous SiO2 shell
# -----------------------------
n_silica_units = 450
min_dist = 1.3

silica_atoms = []

def random_point_in_shell(r_inner, r_outer):
    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)

    radius = np.random.uniform(r_inner**3, r_outer**3) ** (1/3)
    return center + radius * direction

def is_far_enough(new_pos, existing_atoms, min_distance):
    for _, pos in existing_atoms:
        if np.linalg.norm(new_pos - pos) < min_distance:
            return False
    return True

attempts = 0
while len(silica_atoms) < n_silica_units * 3 and attempts < 200000:
    attempts += 1

    # Si atom
    si_pos = random_point_in_shell(core_radius, outer_radius)

    if not is_far_enough(si_pos, silica_atoms, min_dist):
        continue

    silica_atoms.append(("Si", si_pos))

    # Two O atoms around each Si, simplified amorphous representation
    for _ in range(2):
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        o_pos = si_pos + 1.62 * direction

        dist_from_center = np.linalg.norm(o_pos - center)
        if core_radius <= dist_from_center <= outer_radius:
            silica_atoms.append(("O", o_pos))

atoms.extend(silica_atoms)

# -----------------------------
# Write XYZ
# -----------------------------
with open(output_file, "w") as f:
    f.write(f"{len(atoms)}\n")
    f.write("CoFe2O4@SiO2 core-shell model for OVITO\n")
    for element, r in atoms:
        f.write(f"{element} {r[0]:.5f} {r[1]:.5f} {r[2]:.5f}\n")

print(f"File created: {output_file}")
print(f"Total atoms: {len(atoms)}")
print("Core: CoFe2O4")
print("Shell: amorphous SiO2")