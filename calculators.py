import os,tempfile
import multiprocessing as mp
import numpy as np
from openbabel import pybel
from ase.calculators.calculator import Calculator, all_changes

from mol import *

import logging
import torch
import torch._dynamo
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from fairchem.core import FAIRChemCalculator
from mace.calculators import mace_mp
torch._dynamo.config.suppress_errors = True

class OpenBabelUFF_serial(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, h=0.01):
        super().__init__()
        self.h = h  # finite difference step (Å)

    def calculate_energy(self, atoms):
        with tempfile.NamedTemporaryFile("w+", suffix=".xyz", delete=False) as tmp:
            write_path = tmp.name
            atoms.write(write_path)
        mol = next(pybel.readfile("xyz", write_path))
        ff = pybel._forcefields["uff"]
        ff.Setup(mol.OBMol)
        energy = ff.Energy() * 0.04336  # kcal/mol → eV
        os.remove(write_path)
        return energy

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        natoms = len(atoms)
        forces = np.zeros_like(positions)
        h = self.h

        energy0 = self.calculate_energy(atoms)

        for i in range(natoms):
            for d in range(3):
                displaced = positions.copy()
                displaced[i, d] += h
                atoms.set_positions(displaced)
                ep = self.calculate_energy(atoms)

                displaced[i, d] -= 2 * h
                atoms.set_positions(displaced)
                em = self.calculate_energy(atoms)

                # central finite difference
                forces[i, d] = -(ep - em) / (2 * h)

        atoms.set_positions(positions)  # restore
        self.results = {
            "energy": energy0,
            "forces": forces
        }

class OpenBabelMMFF94_serial(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, h=0.01):
        super().__init__()
        self.h = h  # finite difference step (Å)

    def calculate_energy(self, atoms):
        with tempfile.NamedTemporaryFile("w+", suffix=".xyz", delete=False) as tmp:
            write_path = tmp.name
            atoms.write(write_path)
        mol = next(pybel.readfile("xyz", write_path))
        ff = pybel._forcefields["mmff94"]
        ff.Setup(mol.OBMol)
        energy = ff.Energy() * 0.04336  # kcal/mol → eV
        os.remove(write_path)
        return energy

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        natoms = len(atoms)
        forces = np.zeros_like(positions)
        h = self.h

        energy0 = self.calculate_energy(atoms)

        for i in range(natoms):
            for d in range(3):
                displaced = positions.copy()
                displaced[i, d] += h
                atoms.set_positions(displaced)
                ep = self.calculate_energy(atoms)

                displaced[i, d] -= 2 * h
                atoms.set_positions(displaced)
                em = self.calculate_energy(atoms)

                # central finite difference
                forces[i, d] = -(ep - em) / (2 * h)

        atoms.set_positions(positions)  # restore
        self.results = {
            "energy": energy0,
            "forces": forces
        }

class OpenBabelUFF(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, h=0.01, nprocs=None):
        super().__init__()
        self.h = h  # finite difference step
        self.nprocs = nprocs or mp.cpu_count()

    def calculate_energy(self, atoms):
        with tempfile.NamedTemporaryFile("w+", suffix=".xyz", delete=False) as tmp:
            write_path = tmp.name
            atoms.write(write_path)
        mol = next(pybel.readfile("xyz", write_path))
        ff = pybel._forcefields["uff"]
        ff.Setup(mol.OBMol)
        energy = ff.Energy() * 0.04336  # kcal/mol → eV
        os.remove(write_path)
        return energy

    def _finite_difference_task(self, args):
        i, d, atoms, h = args
        displaced = atoms.copy()
        pos = displaced.get_positions()

        pos[i, d] += h
        displaced.set_positions(pos)
        ep = self.calculate_energy(displaced)

        pos[i, d] -= 2 * h
        displaced.set_positions(pos)
        em = self.calculate_energy(displaced)

        return i, d, -(ep - em) / (2 * h)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = atoms.copy()
        positions = atoms.get_positions()
        natoms = len(atoms)
        h = self.h

        energy0 = self.calculate_energy(atoms)

        # Prepare tasks: (i, d, atoms, h)
        tasks = [(i, d, atoms, h) for i in range(natoms) for d in range(3)]

        with mp.get_context("spawn").Pool(self.nprocs) as pool:
            results = pool.map(self._finite_difference_task, tasks)

        forces = np.zeros_like(positions)
        for i, d, f in results:
            forces[i, d] = f

        self.results = {
            "energy": energy0,
            "forces": forces
        }

class OpenBabelMMFF94(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, h=0.01, nprocs=None):
        super().__init__()
        self.h = h  # finite difference step
        self.nprocs = nprocs or mp.cpu_count()

    def calculate_energy(self, atoms):
        with tempfile.NamedTemporaryFile("w+", suffix=".xyz", delete=False) as tmp:
            write_path = tmp.name
            atoms.write(write_path)
        mol = next(pybel.readfile("xyz", write_path))
        ff = pybel._forcefields["mmff94"]
        ff.Setup(mol.OBMol)
        energy = ff.Energy() * 0.04336  # kcal/mol → eV
        os.remove(write_path)
        return energy

    def _finite_difference_task(self, args):
        i, d, atoms, h = args
        displaced = atoms.copy()
        pos = displaced.get_positions()

        pos[i, d] += h
        displaced.set_positions(pos)
        ep = self.calculate_energy(displaced)

        pos[i, d] -= 2 * h
        displaced.set_positions(pos)
        em = self.calculate_energy(displaced)

        return i, d, -(ep - em) / (2 * h)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = atoms.copy()
        positions = atoms.get_positions()
        natoms = len(atoms)
        h = self.h

        energy0 = self.calculate_energy(atoms)

        # Prepare tasks: (i, d, atoms, h)
        tasks = [(i, d, atoms, h) for i in range(natoms) for d in range(3)]

        with mp.get_context("spawn").Pool(self.nprocs) as pool:
            results = pool.map(self._finite_difference_task, tasks)

        forces = np.zeros_like(positions)
        for i, d, f in results:
            forces[i, d] = f

        self.results = {
            "energy": energy0,
            "forces": forces
        }

class DihedralBiasCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, dihedrals, bias_type, height, width, h_fd=1e-4):
        super().__init__()
        self.base = base_calc
        self.dihedrals = dihedrals
        self.bias_type = bias_type
        self.height = height
        self.width = width
        self.h_fd = h_fd  # finite difference step for gradients
        self.history = []

    def _bias_energy(self, angle, past_angles=None):
        if self.bias_type == "gaussian":
            e = 0.0
            for past in past_angles:
                delta = (angle - past + np.pi) % (2 * np.pi) - np.pi
                e += self.height * np.exp(-0.5 * (delta / self.width)**2)
            return e
        elif self.bias_type == "harmonic":
            delta = (angle + np.pi) % (2 * np.pi) - np.pi
            return 0.5 * self.height * (delta / self.width)**2
        else:
            return 0.0

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        self.base.calculate(atoms, properties, system_changes)
        base_energy = self.base.results["energy"]
        base_forces = self.base.results.get("forces", np.zeros_like(atoms.positions))

        bias_energy = 0.0
        bias_forces = np.zeros_like(atoms.positions)

        positions = atoms.get_positions()
        past_angles = self.history if self.bias_type == "gaussian" else None

        for dih in self.dihedrals:
            angle = np.deg2rad(get_dihedral_angle(*[positions[i] for i in dih]))
            bias_energy += self._bias_energy(angle, past_angles)

            # Approximate gradient of bias energy via finite differences
            for atom_idx in dih:
                for d in range(3):  # x, y, z
                    displaced = positions.copy()
                    displaced[atom_idx, d] += self.h_fd
                    angle_p = np.deg2rad(get_dihedral_angle(*[displaced[i] for i in dih]))
                    ep = self._bias_energy(angle_p, past_angles)

                    displaced[atom_idx, d] -= 2 * self.h_fd
                    angle_m = np.deg2rad(get_dihedral_angle(*[displaced[i] for i in dih]))
                    em = self._bias_energy(angle_m, past_angles)

                    grad = (ep - em) / (2 * self.h_fd)
                    bias_forces[atom_idx, d] -= grad  # negative gradient

        self.results = {
            "energy": base_energy + bias_energy,
            "forces": base_forces + bias_forces
        }

    def add_hill(self, atoms):
        positions = atoms.get_positions()
        for dih in self.dihedrals:
            angle = np.deg2rad(get_dihedral_angle(*[positions[i] for i in dih]))
            self.history.append(angle)

class HookeanBondCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, bonded_pairs, k, r0_dict):
        super().__init__()
        self.base = base_calc
        self.bonded_pairs = bonded_pairs  # list of (i, j)
        self.k = k  # spring constant (eV/Å²)
        self.r0_dict = r0_dict  # reference bond lengths

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        self.base.calculate(atoms, properties, system_changes)
        base_energy = self.base.results["energy"]
        base_forces = self.base.results.get("forces", np.zeros_like(atoms.positions))

        positions = atoms.get_positions()
        hooke_forces = np.zeros_like(positions)

        # Convert bonded pairs to NumPy array
        bonded_pairs = np.array(self.bonded_pairs, dtype=int)
        i_idx = bonded_pairs[:, 0]
        j_idx = bonded_pairs[:, 1]

        # Get equilibrium distances r0
        r0_array = np.array([self.r0_dict[(i, j)] for i, j in bonded_pairs])

        # Compute displacement vectors
        r_i = positions[i_idx]
        r_j = positions[j_idx]
        r_vec = r_j - r_i
        r = np.linalg.norm(r_vec, axis=1)

        # Avoid division by zero or large dr
        mask = (r > 1e-8) & (np.abs(r - r0_array) < 10.0)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        r_vec = r_vec[mask]
        r = r[mask]
        r0_array = r0_array[mask]
        dr = r - r0_array

        # Compute force magnitudes and force vectors
        f_mag = -self.k * dr
        f_vec = (f_mag / r)[:, None] * r_vec

        # Accumulate forces
        np.add.at(hooke_forces, i_idx, -f_vec)
        np.add.at(hooke_forces, j_idx, +f_vec)

        # Total Hookean energy
        hooke_energy = 0.5 * self.k * np.sum(dr**2)

        # Final result
        self.results = {
            "energy": base_energy + hooke_energy,
            "forces": base_forces + hooke_forces,
        }

def setup_calc(model):

    if model == "orb":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        orbff = pretrained.orb_d3_v2()
        return ORBCalculator(orbff, device=device)
    elif model == "fair":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return FAIRChemCalculator(hf_hub_filename="uma_sm.pt", device=device.type, task_name="omol")
    elif model == "mace":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device.type)
    elif model == 'uff':
        return OpenBabelUFF()
    elif model == 'mmff94':
        return OpenBabelMMFF94()
    elif model == 'uff_serial':
        return OpenBabelUFF_serial()
    elif model == 'mmff94_serial':
        return OpenBabelMMFF94_serial()
    else:
        raise ValueError(f"Unsupported model: {model}")
