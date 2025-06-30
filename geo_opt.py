import argparse,os
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0])
    comment = lines[1].strip()
    atoms = [line.strip() for line in lines[2:] if line.strip()]
    return natoms, comment, atoms

def write_xyz(filename, atoms, energy):
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"Energy = {energy:.8f} Hartree\n")
        for atom in atoms:
            f.write(f"{atom}\n")

def main():
    parser = argparse.ArgumentParser(description="Optimize geometry and output xyz with energy.")
    parser.add_argument("input_xyz", help="Input .xyz file")
    parser.add_argument("--functional", default="wb97x-d3bj", help="DFT functional (default: wb97x-d3bj)")
    parser.add_argument("--basis", default="def2-tzvp", help="Basis set (default: def2-tzvp)")
    parser.add_argument("--output", default="optimized.xyz", help="Output xyz file")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge (default: 0)")
    parser.add_argument("--spin", type=int, default=0, help="2S (number of unpaired electrons, default: 0 for singlet)")
    args = parser.parse_args()

    natoms, comment, atom_lines = read_xyz(args.input_xyz)
    atom_string = '\n'.join(atom_lines)

    mol = gto.Mole()
    mol.atom = atom_string
    mol.basis = args.basis
    mol.charge = args.charge
    mol.spin = args.spin
    mol.verbose = 4
    mol.build()

    # Robust SCF settings
    mf = dft.RKS(mol)
    mf.conv_tol = 1e-6
    mf.max_cycle = 200
    mf.level_shift = 0.3
    mf.damp = 0.2
    mf.init_guess = 'minao'
    mf.xc = args.functional
    
    mol_opt = optimize(mf, maxsteps=1000)

    # Get final energy
    mf_final = dft.RKS(mol_opt)
    mf_final.xc = args.functional
    energy = mf_final.kernel()

    # Prepare xyz lines
    coords = mol_opt.atom_coords() * 0.52917721092  # Convert Bohr to Angstrom
    symbols = [atom[0] for atom in mol_opt._atom]
    xyz_lines = [f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, (x, y, z) in zip(symbols, coords)]

    write_xyz(args.output, xyz_lines, energy)
    print(f"✅ Optimized geometry written to {args.output} with energy {energy:.8f} Hartree")

if __name__ == "__main__":
    main() 