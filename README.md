# MolSnapper: Conditioning Diffusion for Structure Based Drug Design
> This is A tool to condition diffusion model for Generating 3D Drug-Like Molecules.
> 
> This repository is build on [MolDiff](https://proceedings.mlr.press/v202/peng23b.html) code and conditioned MolDiff trained [model](https://github.com/pengxingang/MolDiff/tree/master).


More information can be found in our paper.

![](img.png)
## Installation
### Dependency
The codes have been tested in the following environment:
Package  | Version
--- | ---
Python | 3.9.18
PyTorch | 2.0.1
CUDA | 11.7
PyTorch Geometric | 2.3.1
RDKit | 2022.03.5
Biopython | 1.83
PyTorch Scatter | 2.1.1

### Install via conda yaml file (cuda 11.3)
```bash
conda env create -f env.yaml
conda activate MolSnapper
```

### Install manually
``` bash
conda create -n MolSanpper python=3.9 # optinal, create a new environment
conda activate MolSanpper

conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c pyg pytorch-scatter

# Install other tools
conda install -c conda-forge rdkit
conda install pyyaml easydict python-lmdb -c conda-forge
conda install -c oddt oddt
```


## Dataset
### CrossDocked
Download and the processed testset from DecompDiff repository https://github.com/bytedance/DecompDiff \
Please download the following files:
- test_set.zip
- test_index.pkl

Save them in <test_directory> and process the test data using:
```python
python scripts/prepare_data_cd.py --pairs-paths <test_directory>/test_index.pkl --root-dir <test_directory>  --out-mol-sdf <data_dir>/test_mol.sdf --out-pockets-pkl <data_dir>/test_pockets.pkl --out-table <data_dir>/test_table.csv
```

For example:
```python
python scripts/prepare_data_cd.py --pairs-paths ./../crossdocked/test_index.pkl --root-dir ./../crossdocked/test_set  --out-mol-sdf ./../crossdocked/test_mol.sdf --out-pockets-pkl ./../crossdocked/test_pockets.pkl --out-table ./../crossdocked/test_table.csv
```
#### Processed data
The processed CrossDocked test set can be found in data dir:
``` bash
data
├── crossdocked
│   ├── test_mol.sdf
│   ├── test_pockets.pkl
│   └── test_table.csv
```

### Binding MOAD
Download and split the dataset as described by the authors of DiffSBDD
https://github.com/arneschneuing/DiffSBDD/tree/main \
Save the test set in <test_directory>

After removing water process the test directory using:
```python
python scripts/prepare_moad.py --test_path <test_directory>  --out-mol-sdf <data_dir>/test_mol.sdf --out-pockets-pkl <data_dir>/test_pockets.pkl --out-table <data_dir>/test_table.csv
```
#### Processed data
The processed Binding MOAD data can be found here:
``` bash
data
├── moad
│   ├── test_mol.sdf
│   ├── test_pockets.pkl
│   └── test_table.csv
```
### Raw complex
If you have raw complexes, remove hydrogen and separate the pockets from the ligands using:
```python
python scripts/clean_and_split.py --in-dir <data_directory>  --proteins-dir <pockets_directory> --ligands-dir <ligands_directory>
```

For a given pocket process the pocket
```python
python scripts/prepare_single_complex.py --root_dir  <data_directory>  --ligand_filename <ligand_filename>.sdf  --protein_filename <protein_filename>.pdb --out_pockets_path <output_path>.pkl
```
For example:
```python
python scripts/prepare_single_complex.py --root_dir  <data_directory>  --ligand_filename ligand.sdf --protein_filename data/protein.pdb --out_pockets_path ./data/protein.pkl
```
#### Processed complex
An example od processed complex,(PDB ID: 1h00), can be found here:
``` bash
data
├── example_1h00
│   ├── ref_points.sdf
│   ├── processed_pocket_1h00.pkl
│   └── ligand.sdf
```

## Sample

MolDiff provided the pretrained models, please first download the pretrained model weights from [here](https://drive.google.com/drive/folders/1zTrjVehEGTP7sN3DB5jaaUuMJ6Ah0-ps?usp=sharing) and put them in the `./ckpt` folder. MolSnapper uses the following model weight files: 
- `MolDiff.pt`: the pretrained complete MolDiff model.
- `bond_predictor.pt`: the pretrained bond predictor that is used for bond guidance during sampling.


### Sample molecules for a given pocket
After setting the correct model weight paths in the config file, you can run the following command to sample molecules:

```python
python scripts/sample_single_pocket.py --outdir .<output_directory> --config <path_to_config_file> --device <device_id> --batch_size <batch_size> --pocket_path <pocket_path>.pkl --sdf_path <sdf_path>.sdf --use_pharma <use_pharma> --num_pharma_atoms <num_pharma_atoms> --clash_rate <clash_rate>
```

The parameters are:
- `outdir`: the root directory to save the sampled molecules.
- `config`: the path to the config file.
- `device`: the device to run the sampling.
- `batch_size`: the batch size for sampling. If set to 0 (default), it will use the batch size specified in the config file.
- `pocket_path`: the path to the pocket file (pkl).
- `mol_size`: the size of the generated molecule. 
- `sdf_path`: path to the SDF file that represent either ligands or reference points with atom positions and types.
- `use_pharma`: A boolean parameter indicating whether to extract pharmacophore points from the SDF file or use the SDF as reference points.
- `pharma_th`: determines the minimum percentage of satisfied pharmacophore points required for a generated molecule to be considered valid during the sampling process.
- `clash_rate`: controls the strength of avoiding clashes during the molecule sampling process.
- `distance_th sets`: the threshold for determining whether a pharmacophore is satisfied.
An example command is:
```python
python scripts/sample_single_pocket.py --outdir ./outputs --config ./configs/sample/sample_MolDiff.yml --batch_size 32 --pocket_path ./data/example_1h00/processed_pocket_1h00.pkl --sdf_path ./data/example_1h00/ref_points.sdf --use_pharma False --clash_rate 0.1
```


After sampling, there will be two directories in the `outdir` folder that contains the meta data and the sdf files of the sampling, respectively.

### Sample molecules for all pockets in the test set
 For sample molecules for all the test set use:

```python
python scripts/sample.py --outdir .<output_directory> --config <path_to_config_file> --device <device_id> --batch_size <batch_size> --pocket_dir <data_directory> --num_pharma_atoms <num_pharma_atoms> --clash_rate <clash_rate>

```
An example command is:
```python
python scripts/sample.py --outdir ./outputs --config ./configs/sample/sample_MolDiff.yml --batch_size 32 --pocket_dir ./data/crossdocked  --num_pharma_atoms 20 --clash_rate 0.1
```

## Evaluate
Filter the generted molecules using [PoseBusters](https://github.com/maabuu/posebusters)
To evaluate the generated molecules, run the following command:
```python
python scripts/evaluate.py  <gen_root> --protein_path <protein_path>.pdb --reflig_path <reflig_path> --save_path <save_path>
```
The parameters are:
- `gen_root`: the directory of the sampled molecules.
- `protein_path`: the path to the protein (PDB format).
- `reflig_path`: the path to reference ligand to evaluate similarity (default is None).
- `save_path`: the path directory to save the evaluation results.

For example:
```python
python scripts/evaluate.py  ./outputs/my_run --protein_path ./data/example_1h00/pocket/1h00_protein.pdb --reflig_path ./data/example_1h00/ligand.sdf --save_path ./outputs/my_run/eval 
```


