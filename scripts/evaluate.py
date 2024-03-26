import argparse
import os
import sys
import torch
sys.path.append('.')
import os
from multiprocessing import Pool
from functools import partial
from glob import glob
from utils.misc import *
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
    else:
        reflig = None

    for file_path in mols_paths:
        mol = Chem.SDMolSupplier(file_path)[0]
        # Perform similarity evaluation (if reference ligand is provided)
        if reflig:
            mol = Chem.SDMolSupplier(file_path)[0]
            similarity_results = (calc_SuCOS_normalized(reflig, mol))

        # Perform interaction analysis
        # interaction_results = []
        interaction_results = (get_h_bond(protein_path, file_path))


        scoring_results = (scoring_func.get_chem(mol))

        results.append({
            # **sample_dict,
            'chem_results': scoring_results,
            'interaction': interaction_results,
            'similarity': similarity_results
        })
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('gen_root', type=str)
    parser.add_argument('--protein_path', type=str)
    parser.add_argument('--reflig_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')

    args = parser.parse_args()
    logger = get_logger('sample', args.gen_root)
    logger.info(args)

    gen_dir = args.gen_root
    prb_files = glob(os.path.join(gen_dir, '*shifted.sdf'))

    results = evaluate_single_pocket(mols_paths = prb_files, protein_path = args.protein_path, reflig_path = args.reflig_path)
    if not os.path.exists(args.save_path):
        # Create the directory if it does not exist
        os.makedirs(args.save_path)

    if args.save_path:
        torch.save(results, os.path.join(args.save_path, f'eval_all.pt'))

    qed = [x['chem_results']['qed'] for x in results]
    sa = [x['chem_results']['sa'] for x in results]
    SuCOS_sim = [x['similarity'] for x in results]
    interaction = [len(x['interaction']) for x in results]

    logger.info('QED:   Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(qed), np.median(qed), np.std(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(sa), np.median(sa), np.std(sa)))
    logger.info('SuCOS_sim:    Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(SuCOS_sim), np.median(SuCOS_sim), np.std(SuCOS_sim)))
    logger.info('# interaction:    Mean: %.3f Median: %.3f Std: %.3f' % (np.mean((interaction)), np.median((interaction)), np.std((interaction))))






