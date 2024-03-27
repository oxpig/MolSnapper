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


def process_pdb_files(input_dir, proteins_dir,):
    os.makedirs(proteins_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir)):
        if fname.endswith('.pdb'):
            pdb_code = fname.split('.')[0]
            input_path = os.path.join(input_dir, fname)
            temp_path_0 = os.path.join(proteins_dir, f'{pdb_code}_temp_0.pdb')

            out_protein_path = os.path.join(proteins_dir, f'{pdb_code}_protein.pdb')
            # Extracts one or more models from a PDB file.
            # subprocess.run(f'pdb_selmodel -1 {input_path} > {temp_path_0}', shell=True)
            # Deleting hydrogens (water)
            subprocess.run(f'pdb_delelem -H {input_path} > {temp_path_0}', shell=True)
            # Removes all HETATM records in the PDB file. (only the protein remains)
            subprocess.run(f'pdb_delhetatm {temp_path_0} > {out_protein_path}', shell=True)


            os.remove(temp_path_0)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', action='store', type=str, required=True)
    parser.add_argument('--proteins-dir', action='store', type=str, required=True)
    args = parser.parse_args()
    disable_rdkit_logging()
    process_pdb_files(input_dir=args.in_dir, proteins_dir=args.proteins_dir)
