from rdkit import Chem

ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
DATA_ATTRS_TO_PAD = {
    'positions', 'one_hot',  'lig_mask', 'pocket_mask', 'fixed_atom_mask','ligand_centroids'
}

def create_dummy_molecule(coords):
    # Create an RDKit molecule object
    mol = Chem.RWMol()
    # Add dummy atoms to the molecule using the provided coordinates
    for coord in coords:
        x, y, z = coord
        atom = Chem.Atom(0)  # 0 represents a dummy atom
        atom.SetNoImplicit(True)  # Set to avoid implicit valence errors
        atom_idx = mol.AddAtom(atom)  # Add the atom to the molecule
    conf = Chem.Conformer(len(coords))
    for i, coord in enumerate(coords):
        x, y, z = coord
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))  # Set atom position
    mol.AddConformer(conf, assignId=True)  # Add conformer to the molecule
    return mol