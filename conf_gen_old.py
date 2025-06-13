import argparse
import os
import multiprocessing as mp
from ase.io import read, write

from calculators import *
from dynamics import *
from optimizer import *
from mol import *

def run_single_md(run_id, atoms, steps, temperature, calculator, traj_dir):
    from copy import deepcopy
    import numpy as np

    atoms = deepcopy(atoms)
    traj_file = os.path.join(traj_dir, f"md_run_{run_id}.traj")
    xyz_file = os.path.join(traj_dir, f"md_run_{run_id}.xyz")
    log_file = os.path.join(traj_dir, f"md_run_{run_id}.log")

    # Add small random noise to diversify starting geometries
    noise = 0.05 * (2 * np.random.rand(*atoms.positions.shape) - 1)
    atoms.set_positions(atoms.get_positions() + noise)

    run_highT_md(
        atoms=atoms,
        steps=steps,
        temperature=temperature,
        timestep=1,
        traj_file=traj_file,
        log_file=log_file,
        xyz_file=xyz_file,
        calculator=calculator,
        dihedrals=get_dihedrals(atoms),
        bias_type="gaussian",
        hill_height=0.02,
        hill_width=0.3,
        hill_interval=5
    )

    return xyz_file

def conformer_search(input_file, steps, temperature, n_confs, out_xyz, model, n_walkers):
    
    atoms = read(input_file)

    traj_dir = "data"
    os.makedirs(traj_dir, exist_ok=True)

    calculator = setup_calc(model)
    print(f"Using calculator: {model}")

    # Step 1: Optimize initial geometry
    print("Optimizing initial structure...")
    atoms, initial_energy = optimize_geometry(atoms, calculator)
    write(os.path.join(traj_dir,"initial_optimized.xyz"), atoms)

    # Step 2: Run parallel MD simulations
    print(f"Running {n_walkers} parallel high-T MD simulations...")
    calculator = setup_calc('uff_serial')  # Use serial UFF for MD to avoid issues with multiprocessing
    args = [(i, atoms, steps, temperature, calculator, traj_dir) for i in range(n_walkers)]
    with mp.Pool(processes=n_walkers) as pool:
        xyz_files = pool.starmap(run_single_md, args)

    # Step 3: Merge all XYZ files into one master
    print("Merging all XYZ files into master_trajectory.xyz...")
    master_xyz_path = os.path.join(traj_dir, "master_trajectory.xyz")
    all_traj = []
    with open(master_xyz_path, "w") as master_file:
        for xyz_file in xyz_files:
            if os.path.exists(xyz_file):
                frames = read(xyz_file, index=":")
                all_traj.extend(frames)
                write(master_file, frames, format="xyz")
    print(f"{len(all_traj)} total frames written to {master_xyz_path}")

    # Step 4: Cluster conformers
    print("Clustering conformers...")
    clustered = cluster_conformers(all_traj, threshold=0.75)
    print(f"{len(clustered)} unique conformers identified.")

    # Step 5: Optimize clustered conformers
    print("Optimizing conformers...")
    calculator = setup_calc(model)
    optimized = optimize_conformers(clustered, calculator)

    # Step 6: Filter and save best conformers
    filtered = [a for a in optimized if a.info["energy"] < initial_energy]
    print(f"{len(filtered)} conformers have energy lower than the optimized starting structure.")
    if len(filtered) == 0:
        print("No better conformers found. Best conformer is in initial_optimized.xyz.")
        return
    filtered = sorted(filtered, key=lambda a: a.info["energy"])[:n_confs]
    for atom in filtered:
        for key in ["energy", "forces", "stress"]:
            atom.info.pop(key, None)
    out_xyz = os.path.join(traj_dir, out_xyz)
    write(out_xyz, filtered, write_results=False)
    print(f"Saved {len(filtered)} better conformers to {out_xyz}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel conformer search with metadynamics")
    parser.add_argument("input_file", help="Input .xyz file with molecular structure")
    parser.add_argument("--model", choices=["orb", "fair", "mace", "uff", "mmff94"], default="uff", help="Force field or ML model")
    parser.add_argument("--steps", type=int, default=1000, help="MD steps per walker")
    parser.add_argument("--temperature", type=float, default=1000, help="Temperature in K for metadynamics")
    parser.add_argument("--n_confs", type=int, default=10, help="Max number of conformers to retain")
    parser.add_argument("--n_walkers", type=int, default=5, help="Number of parallel MD walkers")
    parser.add_argument("--out_xyz", default="ase_conformers.xyz", help="Output XYZ file with filtered conformers")
    args = parser.parse_args()

    # When running using MP, daemonic processes are not allowed to have children.
    if args.model in ["uff", "mmff94"]:
        args.model += "_serial"

    conformer_search(
        input_file=args.input_file,
        steps=args.steps,
        temperature=args.temperature,
        n_confs=args.n_confs,
        out_xyz=args.out_xyz,
        model=args.model,
        n_walkers=args.n_walkers
    )