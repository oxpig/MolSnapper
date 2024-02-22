import os.path
from collections import defaultdict
from typing import Iterable, Dict, List, Tuple

import numpy as np
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem


PHARMACOPHORE_FAMILES_TO_KEEP = ('Donor', 'Acceptor')
FAMILY_MAPPING = {'Donor': 1, 'Acceptor': 2}
_FEATURES_FACTORY = []

def get_features_factory(features_names, resetPharmacophoreFactory=False):
    global _FEATURES_FACTORY, _FEATURES_NAMES
    if resetPharmacophoreFactory or (len(_FEATURES_FACTORY) > 0 and _FEATURES_FACTORY[-1] != features_names):
        _FEATURES_FACTORY.pop()
        _FEATURES_FACTORY.pop()
    if len(_FEATURES_FACTORY) == 0:
        feature_factory = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        _FEATURES_NAMES = features_names
        if features_names is None:
            features_names = list(feature_factory.GetFeatureFamilies())

        _FEATURES_FACTORY.extend([feature_factory, features_names])
    return _FEATURES_FACTORY


def getPharamacophoreCoords(mol, features_names: Iterable[str] = PHARMACOPHORE_FAMILES_TO_KEEP, confId:int=-1) -> \
        Tuple[Dict[str, np.ndarray],  Dict[str, List[np.ndarray]]] :

    feature_factory, keep_featnames = get_features_factory(features_names)
    rawFeats = feature_factory.GetFeaturesForMol(mol, confId=confId)
    featsDict = defaultdict(list)
    idxsDict = defaultdict(list)
    allCoords = defaultdict(list)

    for f in rawFeats:
        if f.GetFamily() in keep_featnames:
            featsDict[f.GetFamily()].append(np.array(f.GetPos(confId=f.GetActiveConformer())))
            idxsDict[f.GetFamily()].append(np.array(f.GetAtomIds()))
            allCoords['Coords'].append(np.array(f.GetPos(confId=f.GetActiveConformer())))
            allCoords['Family'].append(f.GetFamily())

    new_feats_dict = {}
    for key in featsDict:
        new_feats_dict[key] = np.concatenate(featsDict[key]).reshape((-1,3))
    return new_feats_dict, idxsDict, allCoords


def checkPharmacophoreSatisfaction(mol, ref_coords, ref_labels,th):
    # Get the features of the new molecule
    _, _, new_feats = getPharamacophoreCoords(mol)
    new_coords = np.array(new_feats['Coords'])
    new_labels = new_feats['Family']
    label_mapping = {'Donor': 1, 'Acceptor': 2}

    # Convert labels to numerical values
    new_labels = np.array([label_mapping[label] for label in new_labels])

    # Calculate the pairwise distances between ref_coords and new_coords
    try:
        dist_matrix = np.linalg.norm(ref_coords[:, np.newaxis, :].cpu() - new_coords, axis=2)
    except:
        return 0

    # Find matching labels and distances within the threshold
    matched_indices = np.where(np.equal.outer(ref_labels, new_labels) & (dist_matrix <= th))
    # matched_indices = np.where(np.equal.outer(ref_labels, new_labels) & (dist_matrix <= th))

    # Count the number of satisfied constraints
    satisfied_count = len(set(matched_indices[0]))

    fraction_satisfied = satisfied_count / len(ref_coords)
    print(fraction_satisfied)
    return fraction_satisfied