import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
from rdkit.DataStructs import TanimotoSimilarity

fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder',
        'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')

def get_FeatureMapScore(small_m, large_m, score_mode=FeatMaps.FeatMapScoreMode.Best):
    featLists = []
    for m in [small_m, large_m]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = score_mode
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))
    return fm_score

def calc_SuCOS_normalized(reflig, prb_mol):
    fm_score = get_FeatureMapScore(reflig, prb_mol)
    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(reflig, prb_mol,
                                                     allowReordering=False)
    SuCOS_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)
    return SuCOS_score

def get_similarity(reflig, mol):
    mol_finger = Chem.RDKFingerprint(mol)
    reflig_finger = Chem.RDKFingerprint(reflig)
    return TanimotoSimilarity(mol_finger, reflig_finger)