import numpy as np
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.io import write
from ase.calculators.calculator import Calculator, all_changes

from calculators import *

def run_highT_md(
    atoms,
    steps,
    temperature,
    timestep,
    traj_file,
    calculator,
    log_file="md.log",
    xyz_file="md.xyz",
    dihedrals=None,
    bias_type="gaussian",  # or "harmonic"
    hill_height=0.02,
    hill_width=0.3,
    hill_interval=1
    ):

    atoms = atoms.copy()
    atoms.calc = calculator

    if dihedrals:
        print(f"Applying {bias_type} bias on {len(dihedrals)} dihedrals")

        # Wrap calculator
        
        dihcalc = DihedralBiasCalculator(
            base_calc=calculator,
            dihedrals=dihedrals,
            bias_type=bias_type,
            height=hill_height,
            width=hill_width
        )
        
        r0_dict = {}                                                                                                                                                                                               
        for i, j in get_bonds(atoms):
            r0_dict[(i, j)] = atoms.get_distance(i, j)

        atoms.calc = HookeanBondCalculator(
            base_calc=dihcalc,
            bonded_pairs=get_bonds(atoms),
            k=5.0,
            r0_dict=r0_dict
        )
        '''
        atoms.calc = DihedralBiasCalculator(
            base_calc=calculator,
            dihedrals=dihedrals,
            bias_type=bias_type,
            height=hill_height,
            width=hill_width
        )
        '''
    # Initialize MD
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = Langevin(atoms, timestep * fs, temperature_K=temperature, friction=0.01)

    # Attach trajectory and logger
    traj = Trajectory(traj_file, 'w', atoms)
    dyn.attach(traj.write, interval=1)

    if xyz_file:
        print(f"Saved trajectory to {traj_file} and {xyz_file}, log in {log_file}")
        def write_xyz():
            write(xyz_file, atoms, append=True)
        dyn.attach(write_xyz, interval=1)

    logfile = open(log_file, "w")
    logger = MDLogger(dyn, atoms, logfile, header=True, peratom=False)
    dyn.attach(logger, interval=1)

    # Metadynamics hill deposition
    if dihedrals and bias_type == "gaussian":
        def deposit_hill():
            #atoms.calc.add_hill(atoms)
            atoms.calc.base.add_hill(atoms)
            #print(f"[Step {dyn.nsteps}] Hill added")
        dyn.attach(deposit_hill, interval=hill_interval)
    
    print(f"Running MD for {steps} steps at {temperature} K...")
    dyn.run(steps)

    # Save trajectory
    traj.close()
    frames = Trajectory(traj_file)
    
    return traj_file