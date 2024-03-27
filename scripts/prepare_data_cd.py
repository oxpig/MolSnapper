import argparse
import glob
import itertools
import numpy as np
import pandas as pd
import os
import pickle

from rdkit import Chem, Geometry
from tqdm import tqdm
from Bio.PDB import PDBParser

from pdb import set_trace
def get_pocket(mol, pdb_path):
    struct = Chem.MolFromPDBFile(pdb_path, sanitize=False)
    residue_ids = []
    atom_coords = []

    conformer = struct.GetConformer()
    for atom in struct.GetAtoms():
        resid = atom.GetPDBResidueInfo().GetResidueNumber()
        atom_index = atom.GetIdx()
        atom_coords.append(conformer.GetAtomPosition(atom_index))
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

    for i, (atom_coord, resid) in enumerate(zip(atom_coords, residue_ids)):
        # Check if the residue is in contact_residues
        if resid not in contact_residues:
            continue

        # atom_coord is a 3D coordinate
        atom_coord = atom_coord.tolist()

        # Append the atom coordinates and type to the full pocket lists
        atom_type = struct.GetAtomWithIdx(i).GetSymbol()  # Adjust this to get the correct atom type
        pocket_coords_full.append(atom_coord)
        pocket_types_full.append(atom_type)

        # Check if the atom is a backbone atom (N, CA, C, O)
        atom_name = struct.GetAtomWithIdx(i).GetPDBResidueInfo().GetName()
        if atom_name in {'N', 'CA', 'C', 'O'}:
            pocket_coords_bb.append(atom_coord)
            pocket_types_bb.append(atom_type)

    return {
        'full_coord': pocket_coords_full,
        'full_types': pocket_types_full,
        'bb_coord': pocket_coords_bb,
        'bb_types': pocket_types_bb,
    }

def get_pocket_bio(mol, pdb_path):
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


def process_sdf(root_dir,pairs_paths):

    molecules = []
    pockets = []
    out_table = []
    uuid = 0

    for file_dict in pairs_paths:
        print(uuid)
        pdb_path = file_dict.get("src_protein_filename")
        sdf_path = file_dict.get("src_ligand_filename")
        mol = Chem.SDMolSupplier(os.path.join(root_dir, sdf_path))[0]
        if mol is None:
            print(sdf_path)
            continue
        try:
            pocket = get_pocket(mol, os.path.join(root_dir,pdb_path))
        except:
            pocket = get_pocket_bio(mol, os.path.join(root_dir, pdb_path))
            print(f'bio pocket {uuid}')
        molecules.append(mol)
        pockets.append(pocket)
        head, tail = os.path.split(sdf_path)
        mol_name = tail[:-4]
        out_table.append({
            'uuid': uuid,
            'molecule_name': mol_name,
            'pocket_full_size': len(pocket['full_types']),
            'pocket_bb_size': len(pocket['bb_types']),
            'molecule_size': mol.GetNumAtoms(),
        })

        uuid += 1

    return molecules, pockets, pd.DataFrame(out_table)


def run(
        pairs_paths,
        root_dir,
        out_mol_path,
        out_pockets_path,
        out_table_path,
        progress=True
):
    with open(pairs_paths, "rb") as file:
        pairs = pickle.load(file)
    molecules, pockets, out_table = process_sdf(root_dir,pairs)

    bad_idx = set()
    dummy_smiles = 'OC(C1CCC(CS)CC1)N1CCC(CC2CCCCC2)CC1'
    with Chem.SDWriter(open(out_mol_path, 'w')) as writer:
        for i, mol in enumerate(molecules):
            try:
                writer.write(mol)
            except:
                bad_idx.add(i)
                writer.write(Chem.MolFromSmiles(dummy_smiles))  # Dummy mol that will be filtered out
    # Dummy mol that will be filtered out
    with open(out_pockets_path, 'wb') as f:
        pickle.dump(pockets, f)

    # Writing bad idx
    out_table = out_table.reset_index(drop=True)
    out_table['discard'] = False
    for idx in bad_idx:
        out_table.loc[idx, 'discard'] = True
    out_table.to_csv(out_table_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs-paths', action='store', type=str, required=True)
    parser.add_argument('--root-dir', action='store', type=str, required=True)
    parser.add_argument('--out-mol-sdf', action='store', type=str, required=True)
    parser.add_argument('--out-pockets-pkl', action='store', type=str, required=True)
    parser.add_argument('--out-table', action='store', type=str, required=True)
    args = parser.parse_args()

    run(
        pairs_paths=args.pairs_paths,
        root_dir=args.root_dir,
        out_mol_path=args.out_mol_sdf,
        out_pockets_path=args.out_pockets_pkl,
        out_table_path=args.out_table,
    )