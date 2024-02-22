import numpy as np
import os.path as op
from copy import deepcopy
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed
from rdkit.Chem import rdMolDescriptors
from rdkit.six.moves import cPickle
from rdkit.six import iteritems
import math
# from utils.datasets import get_dataset
from rdkit.Chem.FilterCatalog import *
from collections import Counter
_fscores = None



def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = cPickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro
def calculateScore(m, verbose=False):
    if _fscores is None:
        readFragmentScores()
    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    if verbose:
        print('score1: ', score1)
        print('-sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty: ',
              sizePenalty, stereoPenalty, spiroPenalty, bridgePenalty, macrocyclePenalty)
        print('score2: ', score2)
        print('score3: ', score3)
        print('raw sascore: ', sascore)
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    if verbose:
        print('smooth sascore: ', sascore)
        print('norm sascore: ', round((10 - sascore) / 9, 2))
    return sascore
def compute_sa_score(rdmol):
    rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
    sa = calculateScore(rdmol)
    sa_norm = round((10 - sa) / 9, 2)
    return sa_norm

def is_pains(mol):
    params_pain = FilterCatalogParams()
    params_pain.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog_pain = FilterCatalog(params_pain)
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    entry = catalog_pain.GetFirstMatch(mol)
    if entry is None:
        return False
    else:
        return True


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    all_rules = [int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]]
    return all_rules


def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    try:
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
        for confId in confIds:
            AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
            rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
            rmsd_list.append(rmsd)
        # mol3d = Chem.RemoveHs(mol3d)
        rmsd_list = np.array(rmsd_list)
        return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]
    except:
        return [np.nan, np.nan, np.nan]


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    normalized_sa = sa_score / len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])
    logp_score = get_logp(mol)
    lipinski_score = np.sum(obey_lipinski(mol))
    Chem.GetSymmSSSR(mol)
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    # hacc_score = Lipinski.NumHAcceptors(mol)
    # hdon_score = Lipinski.NumHDonors(mol)

    return {
        'qed': qed_score,
        'sa': sa_score,
        'normalized_sa': normalized_sa,
        'logp': logp_score,
        'lipinski': lipinski_score,
        'ring_size': ring_size
    }
