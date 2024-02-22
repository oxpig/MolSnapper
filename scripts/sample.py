import sys
import shutil
import argparse
import itertools
import json
sys.path.append('.')
# import torch.utils.tensorboard
from utils.dataset_pharama import get_test_dataloader
from models.model import MolDiff
from models.bond_predictor import BondPredictor
from utils.sample import seperate_outputs
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.pocket_sample_utils import *
from utils.utils_pharmacophores import checkPharmacophoreSatisfaction
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))


def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False

def prepare_pharama_bacth(x_values,x_label, batch_numbers, device):
    x = torch.zeros(len(batch_numbers),3)  # Initialize x with zeros
    x_family = torch.zeros(len(batch_numbers))
    end_index = x_values.shape[0]
    prev_batch = None

    for i, batch in enumerate(batch_numbers):
        if batch != prev_batch:
            # Assign values to x starting from the batch boundary
            x[i:i + end_index] = x_values
            x_family[i:i + end_index] = x_label

        prev_batch = batch
    ref_type_one_hot = F.one_hot(x_family.to(torch.int64), 8).float().clamp(min=1e-30)
    return x.to(device), ref_type_one_hot.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample/sample_MolDiff.yml')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--pocket_dir', type=str, default="outputs/pocket")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--pharma_th',type=float,default=0.5)
    parser.add_argument('--clash_rate', type=float, default=0.1)
    parser.add_argument('--num_pharma_atoms', type=int, default=20)
    parser.add_argument('--distance_th', type=int, default=1.)
    args = parser.parse_args()
    args.device = torch.device('cpu')
    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint,map_location=args.device)
    train_config = ckpt['config']

    # # Logging
    log_root = args.outdir.replace('outputs', 'outputs_vscode') if sys.argv[0].startswith('/data') else args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name, tag = 'clash_rate_'+str(args.clash_rate))
    logger = get_logger('sample', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # # Transform
    logger.info('Loading data placeholder...')
    featurizer = FeaturizeMol(train_config.chem.atomic_numbers, train_config.chem.mol_bond_types,
                            use_mask_node=train_config.transform.use_mask_node,
                            use_mask_edge=train_config.transform.use_mask_edge,)
    max_size = None
    add_edge = getattr(config.sample, 'add_edge', None)
    ## Data

    pocket_loader = get_test_dataloader(args.pocket_dir,args.num_pharma_atoms, torch.device('cpu:0'))

    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'diffusion':
        model = MolDiff(
                    config=train_config.model,
                    num_node_types=featurizer.num_node_types,
                    num_edge_types=featurizer.num_edge_types
                ).to(args.device)
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # # Bond predictor adn guidance
    if 'bond_predictor' in config:
        logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(ckpt_bond['config']['model'],
                featurizer.num_node_types,
                featurizer.num_edge_types-1 # note: bond_predictor not use edge mask
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
    else:
        bond_predictor = None
    if 'guidance' in config.sample:
        guidance = config.sample.guidance  # tuple: (guidance_type[entropy/uncertainty], guidance_scale)
    else:
        guidance = None

    with open(os.path.join(args.pocket_dir, 'filetered_uuid.json'),
              'r') as file:
        filetered_uuid = json.load(file)
    # # generating molecules

    for i, data in enumerate(pocket_loader):
        if data['uuid'][0] in filetered_uuid:
            pool = EasyDict({
                'failed': [],
                'finished': [],
                'score': []
            })

            uuid = data['uuid'][0]
            family_labels = data['family_labels'][0]
            pharamacophore_coords = data['pharamacophore_coords'][0]
            pocket_x = data['pocket_x'][0]
            mean = pharamacophore_coords.mean(0)
            if pharamacophore_coords.shape[0] == 1:
                mean = mean+torch.randn_like(mean)
            pocket_x = (pocket_x - mean).to(args.device)
            pharamacophore_orig = pharamacophore_coords
            pharamacophore_coords = (pharamacophore_coords - mean).to(args.device)
            num_atoms = data['num_atoms'][0]

            while len(pool.finished) < config.sample.num_mols:
                if len(pool.failed) > 3 * (config.sample.num_mols):
                    logger.info('Too many failed molecules. Stop sampling.')
                    break

                # prepare batch
                batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
                n_graphs = min(batch_size, (config.sample.num_mols - len(pool.finished))*2)
                batch_holder = make_data_placeholder(n_graphs=n_graphs, device=args.device, mean_size=num_atoms)
                batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge']
                ref_coords,ref_type_one_hot = prepare_pharama_bacth(pharamacophore_coords,family_labels, batch_node,args.device)
                ref_mask = (ref_coords.sum(dim=-1) != 0).to(args.device)
                # inference
                outputs = model.sample_in_pocket(
                    n_graphs=n_graphs,
                    batch_node=batch_node,
                    halfedge_index=halfedge_index,
                    batch_halfedge=batch_halfedge,
                    bond_predictor=bond_predictor,
                    guidance=guidance,
                    ref_coords=ref_coords,
                    ref_mask=ref_mask,
                    ref_type=ref_type_one_hot,
                    log_dir=log_dir,
                    condition_clash=True,
                    pocket_x=pocket_x,
                    clash_rate=args.clash_rate
                )
                
                outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}

                # decode outputs to molecules
                batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
                try:
                    output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
                except:
                    continue
                gen_list = []
                Pharma_score = []

                for i_mol, output_mol in enumerate(output_list):
                    mol_info = featurizer.decode_output(
                        pred_node=output_mol['pred'][0],
                        pred_pos=output_mol['pred'][1],
                        pred_halfedge=output_mol['pred'][2],
                        halfedge_index=output_mol['halfedge_index'],
                    )  # note: traj is not used

                    try:
                        rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
                        mol_info_shifted = mol_info
                        mol_info_shifted['atom_pos'] = mol_info_shifted['atom_pos'] + mean.cpu().numpy()
                        rdmol_shifted = reconstruct_from_generated_with_edges(mol_info_shifted, add_edge=add_edge)
                    except MolReconsError:
                        pool.failed.append(mol_info)
                        logger.warning('Reconstruction error encountered.')
                        continue
                    mol_info['rdmol'] = rdmol
                    mol_info['rdmol_shifted'] = rdmol_shifted
                    smiles = Chem.MolToSmiles(rdmol)
                    mol_info['smiles'] = smiles

                    correctPharmacophore =  checkPharmacophoreSatisfaction(rdmol, pharamacophore_coords,family_labels, args.distance_th)

                    if '.' in smiles:
                        logger.warning('Incomplete molecule: %s' % smiles)
                        pool.failed.append(mol_info)

                    elif correctPharmacophore < args.pharma_th:
                        logger.warning('Pharmacophore %f ' % correctPharmacophore)
                        pool.failed.append(mol_info)

                    else:   # Pass checks!
                        logger.info('Success: %s' % smiles)
                        gen_list.append(mol_info)
                        Pharma_score.append(correctPharmacophore)
                        # pool.finished.append(mol_info)

                # # Save sdf mols
                sdf_dir = log_dir + '_SDF'
                os.makedirs(sdf_dir, exist_ok=True)
                os.makedirs(os.path.join(sdf_dir,str(uuid)), exist_ok=True)
                mol = create_dummy_molecule(pharamacophore_coords.cpu().numpy())
                with Chem.SDWriter(open(os.path.join(sdf_dir,str(uuid), 'pharmacophore_coords.sdf'), 'w')) as writer:
                    writer.write(mol)

                mol = create_dummy_molecule(pharamacophore_orig.cpu().numpy())
                with Chem.SDWriter(open(os.path.join(sdf_dir, str(uuid), 'pharmacophore_orig.sdf'), 'w')) as writer:
                    writer.write(mol)



                with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
                    for i, data_finished in enumerate(gen_list):
                        smiles_f.write(data_finished['smiles'] + '\n')
                        rdmol = data_finished['rdmol']
                        Chem.MolToMolFile(rdmol, os.path.join(sdf_dir,str(uuid), '%d.sdf' % (i+len(pool.finished))))
                        rdmol_shifted = data_finished['rdmol_shifted']
                        Chem.MolToMolFile(rdmol_shifted, os.path.join(sdf_dir, str(uuid), '%d_shifted.sdf' % (i + len(pool.finished))))

                pool.finished.extend(gen_list)
                pool.score.extend(Pharma_score)
                print_pool_status(pool, logger)

            try:
                with open(os.path.join(sdf_dir,str(uuid), 'correctPharmacophore.txt'), 'a') as writer:
                    for item in pool.score:
                        writer.write(str(item) + '\n')

            except:
                print('No finished mols')

            with open(os.path.join(log_dir, 'pharmacophore_family.txt'), 'a') as writer:
                writer.write('pharmacophore mean' + str(mean.cpu().numpy()) + '\n')
                for label in family_labels:
                    if label == 1:
                        writer.write('Donor\n')
                    elif label == 2:
                        writer.write('Acceptor\n')

    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
    