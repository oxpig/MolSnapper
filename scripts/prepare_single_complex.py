import argparse
import glob
import itertools
import numpy as np
import pandas as pd
import os
import pickle
import argparse
from rdkit import Chem, Geometry
from tqdm import tqdm
from Bio.PDB import PDBParser

from pdb import set_trace
def get_pocket(mol, pdb_path):
    struct = PDBParser().get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []

    for residue in struct.get_residues():
        resid = residue.get_id()[1]
        for atom in residue.get_atoms():
            atom_coords.append(atom.get_coord())
            residue_ids.append(resid)

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    # Check if it is ok
    mol_atom_coords = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    # Check the distance
    contact_residues = np.unique(residue_ids[np.where(distances.min(1) <= 6)[0]])

    pocket_coords_full = []
    pocket_types_full = []

    pocket_coords_bb = []
    pocket_types_bb = []

    for residue in struct.get_residues():
        resid = residue.get_id()[1]
        if resid not in contact_residues:
            continue

        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            atom_type = atom.element.upper()
            atom_coord = atom.get_coord()

            pocket_coords_full.append(atom_coord.tolist())
            pocket_types_full.append(atom_type)

            # Check what does it mean
            if atom_name in {'N', 'CA', 'C', 'O'}:
                pocket_coords_bb.append(atom_coord.tolist())
                pocket_types_bb.append(atom_type)

    return {
        'full_coord': pocket_coords_full,
        'full_types': pocket_types_full,
        'bb_coord': pocket_coords_bb,
        'bb_types': pocket_types_bb,
    }


def process_single_pair(root_dir, ligand_filename, protein_filename):
    mol_path = os.path.join(root_dir, ligand_filename)
    pdb_path = os.path.join(root_dir, protein_filename)

    mol = Chem.SDMolSupplier(mol_path)[0]
    if mol is None:
        print(f"Error loading molecule from {mol_path}")
        return None, None

    pocket = get_pocket(mol, pdb_path)

    data = {
        'pocket_full_size': len(pocket['full_types']),
        'pocket_bb_size': len(pocket['bb_types']),
        'molecule_size': mol.GetNumAtoms(),
    }

    return  pocket, data

def run_single(root_dir, ligand_filename, protein_filename, out_pockets_path):
    pocket, data = process_single_pair(root_dir, ligand_filename, protein_filename)

    if pocket:
        with open(out_pockets_path, 'wb') as f:
            pickle.dump([pocket], f)  # List of one pocket

    else:
        print("Processing failed.")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process ligand-protein pairs and save pockets.')

    # Define command-line arguments
    parser.add_argument('--root_dir', type=str, help='Root directory where input files are located')
    parser.add_argument('--ligand_filename', type=str, help='Filename of the ligand file (SDF format) within the root directory')
    parser.add_argument('--protein_filename', type=str, help='Filename of the protein file (PDB format) within the root directory')
    parser.add_argument('--out_pockets_path', type=str, help='Output path for the processed pockets (pkl format)')


    # Parse command-line arguments
    args = parser.parse_args()

    # Run the function with user-provided arguments
    run_single(
        root_dir=args.root_dir,
        ligand_filename=args.ligand_filename,
        protein_filename=args.protein_filename,
        out_pockets_path=args.out_pockets_path,
    )

if __name__ == "__main__":
    main()