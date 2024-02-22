import oddt
from oddt.interactions import hbonds


def get_h_bond(protein_path, lig_path):
    protein = next(oddt.toolkit.readfile('pdb', protein_path))
    protein.protein = True
    ligand = next(oddt.toolkit.readfile('sdf', lig_path))
    protein_atoms, ligand_atoms, strict = hbonds(protein, ligand)
    formatted_atoms = [f'{resname}-{resnum}' for resname, resnum in
                       zip(protein_atoms['resname'], protein_atoms['resnum'])]
    return formatted_atoms