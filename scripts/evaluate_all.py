import argparse
import os
import sys
import numpy as np
sys.path.append('.')
from utils.similarity import get_similarity, calc_SuCOS_normalized
from multiprocessing import Pool
from functools import partial
from glob import glob
from rdkit import Chem
from utils.misc import *
import pandas as pd
import json
import itertools
from rdkit.DataStructs import TanimotoSimilarity
from utils.scoring_func import compute_sa_score
from itertools import islice
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_similarity(fg_pair):
    return TanimotoSimilarity(fg_pair[0], fg_pair[1])

def get_sim_with_gt(reflig, mol):
        mol_finger = Chem.RDKFingerprint(mol)
        reflig = Chem.RDKFingerprint(reflig)
        return get_similarity((mol_finger, reflig))


def get_diversity( mols, parallel=False):
    fgs = [Chem.RDKFingerprint(mol) for mol in mols]
    all_fg_pairs = list(itertools.combinations(fgs, 2))
    similarity_list = []
    for fg1, fg2 in tqdm(all_fg_pairs, desc='Calculate diversity'):
        similarity_list.append(TanimotoSimilarity(fg1, fg2))

    return 1 - np.mean(similarity_list)

def get_uniqueness(prb_mols):
    smiles = [Chem.MolToSmiles(mol) for mol in prb_mols]

    unique = len(np.unique(smiles)) / len(smiles)
    return unique

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')

class SimilarityAnalysis:
    def __init__(self, data_path, device):

        self.ligands_path = os.path.join(data_path, 'test_mol.sdf')
        self.table_path = os.path.join(data_path, 'test_table.csv')
        self._get_test_mols()

    def _get_test_mols(self):
        table = pd.read_csv(self.table_path)

        self.test_ligands = Chem.SDMolSupplier(self.ligands_path)
        self.test_fingers = [Chem.RDKFingerprint(mol) for mol in self.test_ligands]
        self.test_smiles = [Chem.MolToSmiles(mol) for mol in self.test_ligands]
        self.test_uuid = []
        self.num_atoms = []

        for (_, row) in table.iterrows():
            self.test_uuid.append(row['uuid'])
            self.num_atoms.append(row['molecule_size'])

def eval_single_pocket( reflig, row, args, logger):
    uuid = row['uuid']
    prb_files = glob(os.path.join(args.gen_root, str(uuid), '*shifted.sdf'))
    n_eval_success = 0
    results = []
    prb_mols = []
    if prb_files != []:
        for path in prb_files:
            mol = Chem.SDMolSupplier(path)[0]

            prb_mols.append(mol)

        # chemical and docking check
            try:
                SuCOS_sim = calc_SuCOS_normalized(reflig,mol)
                tanimoto_sim = get_sim_with_gt(reflig,mol)

                n_eval_success += 1
            except Exception as e:
                logger.warning('Evaluation failed for %s' % f'{uuid}')
                print(str(e))
                continue

            results.append({
                'SuCOS_sim': SuCOS_sim,
                'tanimoto_sim': tanimoto_sim,
                'path': path
            })

        diversity = get_diversity(prb_mols)
        unique = get_uniqueness(prb_mols)
        results.append({
            'diversity': diversity,
            'unique':unique
        })
    else:
        return None
    logger.info(f'Evaluate No {uuid} done! {len(prb_files)} samples in total. {n_eval_success} eval success!')
    if args.result_path:
        torch.save(results, os.path.join(args.result_path, f'eval_similarity_{uuid:03d}.pt'))
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_root', type=str)  # 'baselines/results/pocket2mol_pre_dock.pt'
    parser.add_argument('-n', '--eval_num_examples', type=int, default=100)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--data_root', type=str, default='./../Data/CrossDocked/')
    parser.add_argument('--ligands_root', type=str, default='./../Data/CrossDocked/test_mol.sdf')
    parser.add_argument('--table_path', type=str, default='test_table.csv')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--result_path', type=str)

    args = parser.parse_args()

    logger = get_logger('sample', args.gen_root)
    logger.info(args)

    if not os.path.exists(args.result_path):
        # Create the directory if it does not exist
        os.makedirs(args.result_path)

    testset_results = []

    table_path = os.path.join(args.data_root, args.table_path)
    # table_path = os.path.join(args.data_root, 'MOAD_test_table.csv')
    table = pd.read_csv(table_path)

    # Process each subdirectory in parallel using multiprocessing
    test_ligands = Chem.SDMolSupplier(args.ligands_root)

    with Pool(args.num_workers) as p:
        results = (p.starmap(eval_single_pocket,
                                         [(reflig,row, args, logger) for (_, row), reflig in zip(table.iterrows(),test_ligands)]))

        testset_results = [result for result in results if result is not None]

    if args.result_path:
        torch.save(testset_results, os.path.join(args.result_path, f'eval_all.pt'))

    diversity = [r[-1]['diversity'] for r in testset_results]
    unique = [r[-1]['unique'] for r in testset_results]
    SuCOS_sim = [x['SuCOS_sim'] for r in testset_results for x in r[:-1]]
    tanimoto_sim = [x['tanimoto_sim'] for r in testset_results for x in r[:-1]]

    logger.info('diversity:   Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(diversity), np.median(diversity), np.std(diversity)))
    logger.info('unique:   Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(unique), np.median(unique), np.std(unique)))
    logger.info('SuCOS_sim:   Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(SuCOS_sim), np.median(SuCOS_sim), np.std(SuCOS_sim)))
    logger.info('tanimoto_sim:   Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(tanimoto_sim), np.median(tanimoto_sim), np.std(tanimoto_sim)))


    #find top values
    SuCOS_lists = [[x['SuCOS_sim'] for x in r[:-1]] for r in testset_results]
    path_lists = [[x['path'] for x in r[:-1]] for r in testset_results]
    SuCOS_max_values = [max(sublist) for sublist in SuCOS_lists]
    SuCOS_max_paths = [(path_lists[i][sublist.index(max_sa)], max_sa) for i, (sublist, max_sa) in
                    enumerate(zip(SuCOS_lists, SuCOS_max_values))]


    SuCOS_top5_values = [sorted(sublist, reverse=True)[:5] for sublist in SuCOS_lists]
    SuCOS_top5_paths = [[(path_lists[i][sublist.index(value)], value) for value in sorted(sublist, reverse=True)[:5]]
                        for i, sublist in enumerate(SuCOS_lists)]
    SuCOS_top5_sim = [x for r in SuCOS_top5_values for x in r]


    logger.info('Max SuCOS_sim:    Mean: %.3f Median: %.3f Std: %.3f' % (
        np.mean(SuCOS_max_values), np.median(SuCOS_max_values), np.std(SuCOS_max_values)))

    logger.info('SuCOS_top5_means:    Mean: %.3f Median: %.3f Std: %.3f' % (
        np.mean(SuCOS_top5_sim), np.median(SuCOS_top5_sim), np.std(SuCOS_top5_sim)))


    #save the best path
    SuCOS_max_paths_dicts = [{"path": SuCOS_path, "Max SuCOS_sim": max_SuCOS} for SuCOS_path, max_SuCOS in SuCOS_max_paths]
    with open(os.path.join(args.result_path,'SuCOS_max_paths.json'), "w") as json_file:
        json.dump(SuCOS_max_paths_dicts, json_file, indent=4)

    SuCOS_top5_paths_dicts = [{"path": path, "SuCOS_sim": value} for sublist in SuCOS_top5_paths for path, value in
                              sublist]

    with open(os.path.join(args.result_path,'SuCOS_top5_paths.json'), "w") as json_file:
        json.dump(SuCOS_top5_paths_dicts, json_file, indent=4)


    # Adding SA calculation for molecules with SuCOS_top5_values
    results_with_sa = []
    for entry in SuCOS_max_paths_dicts:
            mol_path = entry['path']
            mol = Chem.SDMolSupplier(mol_path)[0]
            sa = compute_sa_score(mol)
            entry['chem_results'] = {'sa': sa}
            results_with_sa.append(entry)

    sa = [x['chem_results']['sa'] for x in results_with_sa]
    logger.info('SA for top SuCOS:   Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(sa), np.median(sa), np.std(sa)))