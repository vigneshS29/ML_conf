from ase.constraints import FixBondLengths
from collections import defaultdict
import numpy as np

UFF_RADII = {
    'H': 0.354, 'He': 0.849, 'Li': 1.336, 'Be': 1.074, 'B': 0.838, 'C': 0.757,
    'N': 0.700, 'O': 0.658, 'F': 0.668, 'Ne': 0.920, 'Na': 1.539, 'Mg': 1.421,
    'Al': 1.244, 'Si': 1.117, 'P': 1.117, 'S': 1.064, 'Cl': 1.044, 'Ar': 1.032,
    'K': 1.953, 'Ca': 1.761, 'Sc': 1.513, 'Ti': 1.412, 'V': 1.402, 'Cr': 1.345,
    'Mn': 1.382, 'Fe': 1.335, 'Co': 1.241, 'Ni': 1.164, 'Cu': 1.302, 'Zn': 1.193,
    'Ga': 1.260, 'Ge': 1.197, 'As': 1.211, 'Se': 1.190, 'Br': 1.192, 'Kr': 1.147,
    'Rb': 2.260, 'Sr': 2.052, 'Y': 1.698, 'Zr': 1.564, 'Nb': 1.473, 'Mo': 1.484,
    'Tc': 1.322, 'Ru': 1.478, 'Rh': 1.332, 'Pd': 1.338, 'Ag': 1.386, 'Cd': 1.403,
    'In': 1.459, 'Sn': 1.398, 'Sb': 1.407, 'Te': 1.386, 'I': 1.382, 'Xe': 1.267,
    'Cs': 2.570, 'Ba': 2.277, 'La': 1.943, 'Hf': 1.611, 'Ta': 1.511, 'W': 1.526,
    'Re': 1.372, 'Os': 1.372, 'Ir': 1.371, 'Pt': 1.364, 'Au': 1.262, 'Hg': 1.340,
    'Tl': 1.518, 'Pb': 1.459, 'Bi': 1.512, 'Po': 1.500, 'At': 1.545, 'Rn': 1.420,
    'default': 0.7
}

def get_bonds(atoms, scale_factor=1.25):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            r_i = UFF_RADII.get(symbols[i], UFF_RADII['default'])
            r_j = UFF_RADII.get(symbols[j], UFF_RADII['default'])
            cutoff = scale_factor * (r_i + r_j)
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < cutoff:
                bonds.append((i, j))
    return bonds

def check_molecular_topology(image,initial):
    # Check if the molecular topology of each image matches the initial structure interms of connectivity
    
    initial_connectivity = get_bonds(initial) 
    initial_connectivity = sorted(initial_connectivity, key=lambda x: (x[0], x[1]))
    image_connectivity = get_bonds(image)
    image_connectivity = sorted(image_connectivity, key=lambda x: (x[0], x[1]))

    if initial_connectivity != image_connectivity:
        return False  
    else:  return True

def fix_bonds(atoms):
    
    bond_list = get_bonds(atoms)
    atoms.set_constraint(FixBondLengths(bond_list))

    print(f"Detected and fixed {len(bond_list)} bonds:")
    for b in bond_list:
        print(f"  Bond between atoms {b[0]} and {b[1]}")

    return atoms

def get_dihedrals(atoms, scale_factor=1.25):
    bonds = get_bonds(atoms, scale_factor)
    neighbors = {i: [] for i in range(len(atoms))}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    dihedrals = set()
    for j in range(len(atoms)):
        for k in neighbors[j]:
            if j >= k:
                continue  # avoid double-counting j–k

            for i in neighbors[j]:
                if i == k:
                    continue
                for l in neighbors[k]:
                    if l == j or l == i:
                        continue
                    dihedrals.add((i, j, k, l))
                    
    print( f"Detected {len(dihedrals)} dihedrals: {dihedrals}" )
    return sorted(dihedrals)

def get_dihedral_angle(p1, p2, p3, p4):
    """Return dihedral angle in degrees between four points."""
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    # Normalize b1 so that it does not influence magnitude of vector
    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))

def infer_cell(atoms, padding=3.0):

    # If the current cell volume is essentially zero, infer a new one
    if atoms.get_cell().volume < 1e-3:
        positions = atoms.get_positions()
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        lengths = maxs - mins

        # Define cell as box with padding
        new_cell = lengths + 2 * padding
        atoms.translate(-mins + padding)  # center the molecule in new cell
        atoms.set_cell(np.diag(new_cell))
        atoms.set_pbc([True, True, True])
    
    return atoms
'''
def get_dihedrals(atoms, bonds):
    neighbors = defaultdict(list)
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    dihedrals = set()

    for j in range(len(atoms)):
        for i in neighbors[j]:
            if i == j:
                continue
            for k in neighbors[j]:
                if k in (i, j):
                    continue
                for l in neighbors[k]:
                    if l in (j, i, k):
                        continue
                    dihedral = (i, j, k, l)
                    dihedrals.add(dihedral)

    return sorted(dihedrals)
'''
def compute_dihedral(atoms, indices):
    pos = atoms.get_positions()
    return get_dihedral_angle(pos[indices[0]], pos[indices[1]], pos[indices[2]], pos[indices[3]])

def get_phi_psi_indices(atoms):
    bonds = get_bonds(atoms)
    neighbors = defaultdict(list)
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Look for N–CA–C–N backbone pattern
    phi = psi = None
    for i in range(len(atoms)):
        if atoms[i].symbol != 'N':
            continue
        n = i
        ca_candidates = [j for j in neighbors[n] if atoms[j].symbol == 'C']
        for ca in ca_candidates:
            ca_neighbors = neighbors[ca]
            c_candidates = [j for j in ca_neighbors if j != n and atoms[j].symbol == 'C']
            for c in c_candidates:
                # Find previous C for φ
                c_prev = [j for j in neighbors[n] if atoms[j].symbol == 'C' and j != ca]
                if c_prev:
                    phi = (c_prev[0], n, ca, c)
                # Find next N for ψ
                n_next = [j for j in neighbors[c] if atoms[j].symbol == 'N' and j != ca]
                if n_next:
                    psi = (n, ca, c, n_next[0])
                if phi and psi:
                    return phi, psi
    return None, None