import numpy as np
import multiprocessing as mp
from ase.optimize import BFGS
from scipy.spatial.distance import pdist, squareform
from ase.optimize import BFGS
from scipy.linalg import orthogonal_procrustes

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

def cluster_RMSD_conformers(images, initial,threshold=0.25):

    initial_coords = initial.get_positions()
    initial_COM = np.mean(initial_coords, axis=0)
    initial_coords = (initial.get_positions() - initial_COM)
    initial_coords = initial_coords.reshape(-1)  # Flatten the initial coordinates
    initial_coords = np.tile(initial_coords, (len(images), 1))  # Repeat initial coordinates for each image
    
    image_coords = np.array([atoms.get_positions() for atoms in images])
    image_COM =  np.mean(image_coords, axis=1)  
    image_coords = image_coords - image_COM[:, np.newaxis, :]  # Center each image around its COM
    image_coords = image_coords.reshape(len(images), -1)  # Flatten each image's coordinates

    #get optimaal rotation matrix for image_coords and initial_coords
    R, _ = orthogonal_procrustes(initial_coords, image_coords)
    rotated_image_coords = image_coords @ R.T  # Rotate the image coordinates
    rmsd_values = np.sqrt(np.mean((rotated_image_coords - initial_coords) ** 2, axis=1))

    selected_indices = np.where(rmsd_values > threshold )[0]
    selected_conformers = [images[i] for i in selected_indices]
    selected_conformers = [i for i in selected_conformers if check_molecular_topology(i, initial)]

    #remove images that are too similar to each other using RMSD also minimize rotation 
    for count_i,i in enumerate(selected_conformers):
        for count_j,j in enumerate(selected_conformers):
            if count_i != count_j:
                coords_i = i.get_positions()
                coords_j = j.get_positions()
                coords_i = coords_i - np.mean(coords_i, axis=0)
                coords_j = coords_j - np.mean(coords_j, axis=0)
                coords_i = coords_i.reshape(-1,1)
                coords_j = coords_j.reshape(-1,1)
                R, _ = orthogonal_procrustes(coords_i, coords_j)
                coords_j = coords_j @ R.T
                # Calculate RMSD after rotation
                rmsd = np.sqrt(np.mean((coords_i - coords_j) ** 2))
                if rmsd < threshold:
                    selected_conformers.remove(j)
    return selected_conformers

def energy_filter(images,energy):
    filtered = []
    seen_energies = set()
    for atoms in images:
        if "energy" in atoms.info and atoms.info["energy"] < energy and np.round(atoms.info["energy"],2) not in seen_energies:
            seen_energies.add(np.round(atoms.info["energy"],2))
            filtered.append(atoms)
        
    return filtered