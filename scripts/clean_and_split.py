import sys
sys.path.append('../../')

import argparse
import os
import subprocess

from rdkit import Chem
from tqdm import tqdm
import numpy as np
from Bio.PDB import PDBParser

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')

def get_relevant_ligands(mol):
    # Finds the disconnected fragments from a molecule
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    ligands = []
    for frag in frags:
        if 10 < frag.GetNumAtoms() <= 40:
            ligands.append(frag)
    return ligands

def process_pdb_files(input_dir, proteins_dir, ligands_dir):
    os.makedirs(proteins_dir, exist_ok=True)
    os.makedirs(ligands_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir)):
        if fname.endswith('.pdb'):
            pdb_code = fname.split('.')[0]
            input_path = os.path.join(input_dir, fname)
            temp_path_0 = os.path.join(proteins_dir, f'{pdb_code}_temp_0.pdb')
            temp_path_1 = os.path.join(proteins_dir, f'{pdb_code}_temp_1.pdb')
            temp_path_2 = os.path.join(proteins_dir, f'{pdb_code}_temp_2.pdb')
            temp_path_3 = os.path.join(proteins_dir, f'{pdb_code}_temp_3.pdb')

            out_protein_path = os.path.join(proteins_dir, f'{pdb_code}_protein.pdb')
            out_ligands_path = os.path.join(ligands_dir, f'{pdb_code}_ligands.pdb')

            # Extracts one or more models from a PDB file.
            subprocess.run(f'pdb_selmodel -1 {input_path} > {temp_path_0}', shell=True)
            # Deleting hydrogens (water)
            subprocess.run(f'pdb_delelem -H {temp_path_0} > {temp_path_1}', shell=True)
            # Removes all HETATM records in the PDB file. (only the protein remains)
            subprocess.run(f'pdb_delhetatm {temp_path_1} > {out_protein_path}', shell=True)

            # Selects all HETATM records in the PDB file.
            subprocess.run(f'pdb_selhetatm {temp_path_1} > {temp_path_2}', shell=True)
            # Deleting hydrogens (water)
            subprocess.run(f'pdb_delelem -H {temp_path_2} > {temp_path_3}', shell=True)
            # ?
            subprocess.run(f'pdb_delelem -X {temp_path_3} > {out_ligands_path}', shell=True)

            os.remove(temp_path_0)
            os.remove(temp_path_1)
            os.remove(temp_path_2)
            os.remove(temp_path_3)

            try:
                # Construct a molecule from a PDB file
                mol = Chem.MolFromPDBFile(out_ligands_path, sanitize=False)
                os.remove(out_ligands_path)
            except Exception as e:
                print(f'Problem reading ligands PDB={pdb_code}: {e}')
                os.remove(out_ligands_path)
                continue

            try:
                ligands = get_relevant_ligands(mol)  # Assuming you have a function like get_relevant_ligands
            except Exception as e:
                print(f'Problem getting relevant ligands PDB={pdb_code}: {e}')
                continue

            for i, lig in enumerate(ligands):
                out_ligand_path = os.path.join(ligands_dir, f'{pdb_code}_{i}.sdf')
                w = Chem.SDWriter(str(out_ligand_path))
                w.write(lig)

def run(input_dir, proteins_dir, ligands_dir):
    os.makedirs(proteins_dir, exist_ok=True)
    os.makedirs(ligands_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir)):
        if fname.endswith('.bio1'):
            pdb_code = fname.split('.')[0]
            input_path = os.path.join(input_dir, fname)
            temp_path_0 = os.path.join(proteins_dir, f'{pdb_code}_temp_0.pdb')
            temp_path_1 = os.path.join(proteins_dir, f'{pdb_code}_temp_1.pdb')
            temp_path_2 = os.path.join(proteins_dir, f'{pdb_code}_temp_2.pdb')
            temp_path_3 = os.path.join(proteins_dir, f'{pdb_code}_temp_3.pdb')

            out_protein_path = os.path.join(proteins_dir, f'{pdb_code}_protein.pdb')
            out_ligands_path = os.path.join(ligands_dir, f'{pdb_code}_ligands.pdb')
            # Extracts one or more models from a PDB file.
            subprocess.run(f'pdb_selmodel -1 {input_path} > {temp_path_0}', shell=True)
            # Deleting hydrogens (water)
            subprocess.run(f'pdb_delelem -H {temp_path_0} > {temp_path_1}', shell=True)
            # Removes all HETATM records in the PDB file. (only the protein remains)
            subprocess.run(f'pdb_delhetatm {temp_path_1} > {out_protein_path}', shell=True)

            # Selects all HETATM records in the PDB file.
            subprocess.run(f'pdb_selhetatm {temp_path_1} > {temp_path_2}', shell=True)
            # Deleting hydrogens (water)
            subprocess.run(f'pdb_delelem -H {temp_path_2} > {temp_path_3}', shell=True)
            #?
            subprocess.run(f'pdb_delelem -X {temp_path_3} > {out_ligands_path}', shell=True)

            os.remove(temp_path_0)
            os.remove(temp_path_1)
            os.remove(temp_path_2)
            os.remove(temp_path_3)

            try:
                # Construct a molecule from a PDB file
                mol = Chem.MolFromPDBFile(out_ligands_path, sanitize=False)
                os.remove(out_ligands_path)
            except Exception as e:
                print(f'Problem reading ligands PDB={pdb_code}: {e}')
                os.remove(out_ligands_path)
                continue

            try:
                ligands = get_relevant_ligands(mol)
            except Exception as e:
                print(f'Problem getting relevant ligands PDB={pdb_code}: {e}')
                continue

            for i, lig in enumerate(ligands):
                out_ligand_path = os.path.join(ligands_dir, f'{pdb_code}_{i}.sdf')
                w = Chem.SDWriter(str(out_ligand_path))
                w.write(lig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', action='store', type=str, required=True)
    parser.add_argument('--proteins-dir', action='store', type=str, required=True)
    parser.add_argument('--ligands-dir', action='store', type=str, required=True)
    args = parser.parse_args()

    disable_rdkit_logging()
    process_pdb_files(input_dir=args.in_dir, proteins_dir=args.proteins_dir, ligands_dir=args.ligands_dir)
