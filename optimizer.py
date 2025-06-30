import numpy as np
import multiprocessing as mp
from ase.optimize import BFGS
from scipy.spatial.distance import pdist, squareform
from ase.optimize import BFGS
from scipy.linalg import orthogonal_procrustes
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

from mol import *

def optimize_geometry(atoms,calculator):
    atoms = infer_cell(atoms)
    atoms.calc = calculator
    print("Running initial geometry optimization...")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.05)
    energy = atoms.get_potential_energy()
    print(f"Geometry optimization complete.")
    print(f"Final optimized energy: {energy:.6f} eV")
    return atoms.copy(),energy

def optimize_single_conformer(args):
    atoms, calculator = args
    atoms = atoms.copy()
    atoms.calc = calculator
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.05)
    energy = atoms.get_potential_energy()
    atoms.info["energy"] = energy
    return atoms

def optimize_geo_DFT(atoms, functional="wb97x-d3bj", basis="def2-tzvp", charge=0, spin=0, verbose=0):

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    atom_lines = [f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, (x, y, z) in zip(symbols, positions)]
    atom_string = "\n".join(atom_lines)

    mol = gto.Mole()
    mol.atom = atom_string
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = verbose
    mol.build()

    mf = dft.RKS(mol)
    mf.conv_tol = 1e-6
    mf.max_cycle = 200
    mf.level_shift = 0.3
    mf.damp = 0.2
    mf.init_guess = 'minao'
    mf.xc = functional

    mol_opt = optimize(mf, maxsteps=1000)

    # Get final energy
    mf_final = dft.RKS(mol_opt)
    mf_final.xc = functional
    energy = mf_final.kernel()

    # Prepare new ASE Atoms object
    coords = mol_opt.atom_coords() * 0.52917721092  # Bohr to Angstrom
    symbols = [atom[0] for atom in mol_opt._atom]
    atoms_opt = Atoms(symbols=symbols, positions=coords)

    return atoms_opt, energy


def optimize_conformers(conformers, calculator, nprocs=None):
    if nprocs is None:
        nprocs = mp.cpu_count()  
    print(f"Optimizing conformers in parallel across {nprocs} CPU cores...")
    args = [(atoms, calculator) for atoms in conformers]
    with mp.get_context("spawn").Pool(processes=nprocs) as pool:
        optimized = pool.map(optimize_single_conformer, args)
    return optimized

def cluster_conformers(images, threshold=0.75):
    coords = np.array([atoms.get_positions().flatten() for atoms in images])
    dmat = squareform(pdist(coords, metric='euclidean'))
    N = len(images)
    picked = []
    for i in range(N):
        if not picked:
            picked.append(i)
        else:
            dists = [dmat[i, j] for j in picked]
            if all(d > threshold for d in dists):
                picked.append(i)
    return [images[i] for i in picked]

def cluster_RMSD_conformers(images, initial, threshold=0.25):
    # Pre-compute initial coordinates once
    initial_coords = initial.get_positions()
    initial_COM = np.mean(initial_coords, axis=0)
    initial_coords = (initial_coords - initial_COM).reshape(-1)
    
    # Pre-allocate arrays for all images
    n_images = len(images)
    image_coords = np.zeros((n_images, len(initial_coords)))
    
    # Process all images at once
    for i, atoms in enumerate(images):
        coords = atoms.get_positions()
        COM = np.mean(coords, axis=0)
        image_coords[i] = (coords - COM).reshape(-1)
    
    # Compute optimal rotation matrix for all images at once
    R, _ = orthogonal_procrustes(np.tile(initial_coords, (n_images, 1)), image_coords)
    rotated_coords = image_coords @ R.T
    
    # Calculate RMSD for all images at once
    rmsd_values = np.sqrt(np.mean((rotated_coords - np.tile(initial_coords, (n_images, 1))) ** 2, axis=1))
    
    # Get indices of conformers above threshold
    selected_indices = np.where(rmsd_values > threshold)[0]
    selected_conformers = [images[i] for i in selected_indices]
    selected_conformers = [i for i in selected_conformers if check_molecular_topology(i, initial)]
    
    # Vectorized pairwise RMSD comparison
    n_selected = len(selected_conformers)
    if n_selected > 1:
        # Pre-allocate arrays for selected conformers
        selected_coords = np.zeros((n_selected, len(initial_coords)))
        for i, atoms in enumerate(selected_conformers):
            coords = atoms.get_positions()
            COM = np.mean(coords, axis=0)
            selected_coords[i] = (coords - COM).reshape(-1)
        
        # Compute pairwise RMSD matrix
        rmsd_matrix = np.zeros((n_selected, n_selected))
        for i in range(n_selected):
            for j in range(i+1, n_selected):
                R, _ = orthogonal_procrustes(selected_coords[i:i+1].T, selected_coords[j:j+1].T)
                rotated_j = selected_coords[j] @ R.T
                rmsd = np.sqrt(np.mean((selected_coords[i] - rotated_j) ** 2))
                rmsd_matrix[i,j] = rmsd_matrix[j,i] = rmsd
        
        # Find unique conformers based on RMSD threshold
        to_keep = np.ones(n_selected, dtype=bool)
        for i in range(n_selected):
            if to_keep[i]:
                to_keep[rmsd_matrix[i] < threshold] = False
                to_keep[i] = True
        
        selected_conformers = [conf for i, conf in enumerate(selected_conformers) if to_keep[i]]
    
    print(f"After filtering based on RMSD {threshold}, {len(selected_conformers)} unique conformers remain.")
    return selected_conformers

def energy_filter(images,energy):
    filtered = []
    seen_energies = set()
    for atoms in images:
        if "energy" in atoms.info and atoms.info["energy"] < energy and np.round(atoms.info["energy"],2) not in seen_energies:
            seen_energies.add(np.round(atoms.info["energy"],2))
            filtered.append(atoms)
        
    return filtered
