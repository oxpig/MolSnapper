<<<<<<< HEAD
import argparse
import os
import sys
import torch
sys.path.append('.')
import os
from multiprocessing import Pool
from functools import partial
from glob import glob
from utils.similarity import get_similarity, calc_SuCOS_normalized
from utils.oddt_interaction import get_h_bond
import utils.scoring_func as scoring_func
from rdkit import Chem
import json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def evaluate_single_pocket( mols_paths, protein_path, reflig_path):
    results = []
    # Load reference ligand molecule (if provided)
    similarity_results = []
    if reflig_path:
        reflig = Chem.SDMolSupplier(reflig_path)[0]

        # Perform similarity evaluation (if reference ligand is provided)
        if reflig:
            for file_path in mols_paths:
                mol = Chem.SDMolSupplier(file_path)[0]
                similarity_results.append(calc_SuCOS_normalized(reflig, mol))

    # Perform interaction analysis

    interaction_results = []
    # for file_path in mols_paths:
    #     interaction_results.append(get_h_bond(protein_path, file_path))
    # Save interaction results (if needed)

    # Perform scoring function evaluation
    scoring_results = []
    for file_path in prb_files:
        mol = Chem.SDMolSupplier(file_path)[0]
        scoring_results.append(scoring_func.get_chem(mol))

    results.append({
        # **sample_dict,
        'chem_results': scoring_results,
        'interaction': interaction_results,
        'similarity': str(similarity_results)
    })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str)
    parser.add_argument('--protein_path', type=str)
    parser.add_argument('--reflig_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')

    args = parser.parse_args()

    gen_dir = args.result_root
    prb_files = glob(os.path.join(gen_dir, '*.sdf'))[:10]

    results = evaluate_single_pocket(mols_paths = prb_files, protein_path = args.protein_path, reflig_path = args.reflig_path)

    if args.save_path:
        torch.save(results, os.path.join(args.save_path, f'eval_all.pt'))

=======
import argparse
import os
import sys
import torch
sys.path.append('.')
import os
from multiprocessing import Pool
from functools import partial
from glob import glob
from utils.similarity import get_similarity, calc_SuCOS_normalized
from utils.oddt_interaction import get_h_bond
import utils.scoring_func as scoring_func
from rdkit import Chem
import json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def evaluate_single_pocket( mols_paths, protein_path, reflig_path):
    results = []
    # Load reference ligand molecule (if provided)
    similarity_results = []
    if reflig_path:
        reflig = Chem.SDMolSupplier(reflig_path)[0]

        # Perform similarity evaluation (if reference ligand is provided)
        if reflig:
            for file_path in mols_paths:
                mol = Chem.SDMolSupplier(file_path)[0]
                similarity_results.append(calc_SuCOS_normalized(reflig, mol))

    # Perform interaction analysis

    interaction_results = []
    # for file_path in mols_paths:
    #     interaction_results.append(get_h_bond(protein_path, file_path))
    # Save interaction results (if needed)

    # Perform scoring function evaluation
    scoring_results = []
    for file_path in prb_files:
        mol = Chem.SDMolSupplier(file_path)[0]
        scoring_results.append(scoring_func.get_chem(mol))

    results.append({
        # **sample_dict,
        'chem_results': scoring_results,
        'interaction': interaction_results,
        'similarity': str(similarity_results)
    })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_root', type=str)
    parser.add_argument('--protein_path', type=str)
    parser.add_argument('--reflig_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')

    args = parser.parse_args()

    gen_dir = args.gen_root
    prb_files = glob(os.path.join(gen_dir, '*.sdf'))[:10]

    results = evaluate_single_pocket(mols_paths = prb_files, protein_path = args.protein_path, reflig_path = args.reflig_path)

    if args.save_path:
        torch.save(results, os.path.join(args.save_path, f'eval_all.pt'))

>>>>>>> b31f05d7b019d58034a328b2c2a19a70bbd22acd
