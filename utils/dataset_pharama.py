import torch
import torch.utils.data as data_utl
import numpy as np
import os
import os.path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from rdkit import Chem
from tqdm import tqdm
import json
from utils.utils_pharmacophores import *
from utils.pocket_sample_utils import *


def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=True) as supplier:
        for molecule in supplier:
            yield molecule


def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot


def parse_molecule(mol):
    one_hot = []
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), ATOM2IDX))

    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot)



class Dataset(Dataset):
    def __init__(self, data_path,num_pharma_atoms, prefix, device):
        if '.' in prefix:
            prefix, pocket_mode = prefix.split('.')
        else:
            parts = prefix.split('_')
            prefix = '_'.join(parts[:-1])
            pocket_mode = parts[-1]

        dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
        # if os.path.exists(dataset_path):
        #     self.data = torch.load(dataset_path, map_location=device)
        # else:
        print(f'Preprocessing dataset with prefix {prefix}')
        self.data = Dataset.preprocess(data_path, num_pharma_atoms,prefix, pocket_mode, device)
        torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path,n_samples, prefix, pocket_mode, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        ligands_path = os.path.join(data_path, f'{prefix}_mol.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(table.iterrows(), read_sdf(ligands_path), pockets),
            total=len(table)
        )
        for (_, row), ligands, pocket_data in generator:
            uuid = row['uuid']

            name = row['molecule_name']
            new_feats_dict, idxsDict, allCoords = getPharamacophoreCoords(ligands)
            try:
                permuted_indices = np.random.permutation(range(len(allCoords['Coords'])))
                pharamacophore_coords = np.array(allCoords['Coords'])[permuted_indices]
                pharamacophore_family = allCoords['Family']
                pharamacophore_coords, unique_indices = np.unique(pharamacophore_coords, axis=0, return_index=True)

                # Convert labels to numerical values
                family_labels = np.array([FAMILY_MAPPING[label] for label in pharamacophore_family])
                family_labels = family_labels[permuted_indices][unique_indices]
                if len(pharamacophore_coords)< n_samples:
                    num_samples = len(pharamacophore_coords)
                else:
                    num_samples = n_samples

                indices = np.random.choice(len(pharamacophore_coords), size=num_samples,
                                           replace=False)
                pharamacophore_coords = pharamacophore_coords[indices]
                family_labels = family_labels[indices]

            except:
                print(str(uuid) +' doesnt have pharamacophore_coords')

            ligands_pos, ligands_one_hot = parse_molecule(ligands)

            # Parsing pocket data
            pocket_pos = pocket_data[f'{pocket_mode}_coord']
            pocket_pos = np.array(pocket_pos)

            positions = np.concatenate([ pocket_pos, ligands_pos], axis=0)


            mask = np.concatenate([
                np.zeros_like(pocket_pos[:,0]),
                np.ones_like(ligands_pos[:,0])
            ])

            pocket_mask = np.concatenate([
                np.ones_like(pocket_pos[:,0]),
                np.zeros_like(ligands_pos[:,0])
            ])


            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=torch.float32, device=device),
                'pocket_mask': torch.tensor(pocket_mask, dtype=torch.float32, device=device),
                'lig_mask': torch.tensor(mask, dtype=torch.float32, device=device),
                'pharamacophore_coords' : torch.tensor(pharamacophore_coords, dtype=torch.float32, device=device),
                'family_labels': torch.tensor(family_labels, dtype=torch.float32, device=device),
                'pocket_x': torch.tensor(pocket_pos, dtype=torch.float32, device=device),
                'num_atoms': len(ligands_pos),
                'unique_indices': len(unique_indices)
            })

        return data

    @staticmethod
    def create_edges(positions, mask):
        ligand_adj = mask[:, None] & mask[None, :]
        proximity_adj = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) <= 6
        full_adj = ligand_adj | proximity_adj
        full_adj &= ~np.eye(len(positions)).astype(bool)
        curr_rows, curr_cols = np.where(full_adj)
        return [curr_rows, curr_cols]


def collate(batch):
    out = {}

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        else:
            continue

    # which is padding
    atom_mask = (out['pocket_mask'].bool() | out['lig_mask'].bool()).to(torch.int8)
    out['atom_mask'] =atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    # In case of MOAD edge_mask is batch_idx
    batch_mask = torch.cat([
        torch.ones(n_nodes, dtype=torch.int8) * i
        for i in range(batch_size)
    ]).to(atom_mask.device)
    out['edge_mask'] = batch_mask

    for key, value in out.items():
        if key in {'lig_mask', 'pocket_mask', 'fixed_atom_mask' }:
            out[key] = out[key][:, :, None]
            continue
        else:
            continue

    return out





def get_test_dataloader(pocket_dir,num_pharma_atoms,device, collate_fn=collate):
    dataset = Dataset(pocket_dir,num_pharma_atoms, 'test_full', device)
    return DataLoader(dataset, 1,  collate_fn=collate_fn, shuffle=False, pin_memory=True)
