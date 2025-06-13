import argparse
import os
import shutil
import multiprocessing as mp
from copy import deepcopy

import numpy as np
from ase.io import read, write

from calculators import *
from dynamics import *
from optimizer import *
from mol import *

def run_single_md(run_id, atoms, steps, temperature, calculator, traj_dir):
    atoms = deepcopy(atoms)
    traj_file = os.path.join(traj_dir, f"md_run_{run_id}.traj")
    xyz_file = os.path.join(traj_dir, f"md_run_{run_id}.xyz")
    log_file = os.path.join(traj_dir, f"md_run_{run_id}.log")

    # Add random perturbation
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


def conformer_search_cyclic(input_file, steps, temperature, n_confs, out_xyz, model, n_walkers, n_cycles):
    atoms = read(input_file)
    root_dir = "data"
    os.makedirs(root_dir, exist_ok=True)

    calculator = setup_calc(model)
    atoms, initial_energy = optimize_geometry(atoms, calculator)
    best_conformers = [atoms]
    best_energy = initial_energy
    write(os.path.join(root_dir, "initial_optimized.xyz"), atoms)

    print(f"Initial energy: {initial_energy:.6f} Eh")

    for cycle in range(n_cycles):
        print(f"\n=== Cycle {cycle + 1} ===")
        cycle_dir = os.path.join(root_dir, f"cycle_{cycle+1}")
        os.makedirs(cycle_dir, exist_ok=True)

        # Run MD in parallel for each seed conformer
        print(f"Running high-T MD from {len(best_conformers)} seed conformers...")

        md_args = []
        md_calc = setup_calc("uff_serial")

        for i, conformer in enumerate(best_conformers):
            for j in range(n_walkers):
                run_id = f"{i}_{j}"
                md_args.append((run_id, conformer, steps, temperature, md_calc, cycle_dir))

        with mp.Pool(processes=min(len(md_args), mp.cpu_count())) as pool:
            xyz_files = pool.starmap(run_single_md, md_args)

        # Merge all XYZs
        print("Merging MD results...")
        master_xyz = os.path.join(cycle_dir, "master_trajectory.xyz")
        all_traj = []
        with open(master_xyz, "w") as master_file:
            for xyz_file in xyz_files:
                if os.path.exists(xyz_file):
                    frames = read(xyz_file, index=":")
                    all_traj.extend(frames)
                    write(master_file, frames, format="xyz")

        print(f"Collected {len(all_traj)} frames.")

        # Cluster
        print("Clustering conformers...")
        clustered = cluster_RMSD_conformers(all_traj,read(input_file), threshold=0.25)
        print(f"{len(clustered)} unique conformers identified.")

        # Optimize
        print("Optimizing clustered conformers...")
        calculator = setup_calc(model)
        optimized = optimize_conformers(clustered, calculator)

        # Re-cluster after optimization
        print("Re-clustering optimized conformers...")
        optimized_clustered = cluster_RMSD_conformers(optimized,read(input_file), threshold=0.25)
        print(f"{len(optimized_clustered)} unique optimized conformers retained after re-clustering.")

        # Filter by energy
        filtered = energy_filter(optimized_clustered, best_energy)
        
        print(f"{len(filtered)} conformers have energy lower than best so far ({best_energy:.6f} Eh).")

        if len(filtered) == 0:
            print("No better conformers found in this cycle.")
            continue

        # Sort and retain top n_confs
        filtered = sorted(filtered, key=lambda a: a.info["energy"])
        best_conformers += filtered

        # Clean and save
        for atom in filtered:
            for key in ["energy", "forces", "stress"]:
                atom.info.pop(key, None)

        cycle_out = os.path.join(cycle_dir, "filtered_conformers.xyz")
        write(cycle_out, filtered, write_results=False)
        print(f"Saved {len(filtered)} new conformers to {cycle_out}")
    
    # Merge all master XYZs
    print("Merging all MD results...")
    all_traj = []
    xyz_files = []
    for cycle in range(cycle + 1):
        xyz_files += [os.path.join(root_dir,os.path.join(f"cycle_{cycle+1}", "master_trajectory.xyz"))]

    master_xyz = os.path.join(root_dir, "all_traj.xyz")
    with open(master_xyz, "w") as master_file:
        for xyz_file in xyz_files:
            if os.path.exists(xyz_file):
                frames = read(xyz_file, index=":")
                all_traj.extend(frames)
                write(master_file, frames, format="xyz")

    print(f"Collected {len(all_traj)} frames.")        
    # Final write of best conformers 
    write( os.path.join(root_dir,out_xyz), best_conformers, write_results=False)
    print(f"Found {len(best_conformers)} low energy conformers after {n_cycles} cycles. Final best conformers saved to {out_xyz}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cyclic parallel conformer search with metadynamics")
    parser.add_argument("input_file", help="Input .xyz file with molecular structure")
    parser.add_argument("--model", choices=["orb", "fair", "mace", "uff", "mmff94"], default="uff", help="Force field or ML model")
    parser.add_argument("--steps", type=int, default=100, help="MD steps per walker")
    parser.add_argument("--temperature", type=float, default=500, help="Temperature in K for metadynamics")
    parser.add_argument("--n_confs", type=int, default=20, help="Top conformers to seed next cycle")
    parser.add_argument("--n_walkers", type=int, default=5, help="Number of MD walkers per conformer")
    parser.add_argument("--n_cycles", type=int, default=2, help="Number of metadynamics cycles")
    parser.add_argument("--out_xyz", default="final_conformers.xyz", help="Final output XYZ file")
    args = parser.parse_args()

    # When running using MP, daemonic processes are not allowed to have children.
    if args.model in ["uff", "mmff94"]:
        args.model += "_serial"

    conformer_search_cyclic(
        input_file=args.input_file,
        steps=args.steps,
        temperature=args.temperature,
        n_confs=args.n_confs,
        out_xyz=args.out_xyz,
        model=args.model,
        n_walkers=args.n_walkers,
        n_cycles=args.n_cycles
    )