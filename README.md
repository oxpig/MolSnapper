# MolSnapper: Conditioning Diffusion for Structure Based Drug Design
> This is A tool to condition diffusion model for Generating 3D Drug-Like Molecules.
> This repository is build on [MolDiff](https://proceedings.mlr.press/v202/peng23b.html) code and train model.


More information can be found in our paper.


## Installation
### Dependency
The codes have been tested in the following environment:
Package  | Version
--- | ---
Python | 3.8.13
PyTorch | 1.10.1
CUDA | 11.3.1
PyTorch Geometric | 2.0.4
RDKit | 2022.03.2


### Install via conda yaml file (cuda 11.3)
```bash
conda env create -f env.yaml
conda activate MolDiff
```

### Install manually

``` bash
conda create -n MolDiff python=3.8 # optinal, create a new environment
conda activate MolDiff

# Install PyTorch (for cuda 11.3)
conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pyg -c pyg

# Install other tools
conda install -c conda-forge rdkit
conda install pyyaml easydict python-lmdb -c conda-forge

# Install tensorboard only for training
conda install tensorboard -c conda-forge
```


## Dataset
### CrossDocked
Download and the processed testset from DecompDiff repository https://github.com/bytedance/DecompDiff \
Please download the following files:
- test_set.zip
- test_index.pkl

Save them in <test_directory> and process the test data using:
```python
python scripts/prepare_cd.py --pairs-paths <test_directory>/test_index.pkl --root-dir <test_directory>  --out-mol-sdf <data_dir>/test_mol.sdf --out-pockets-pkl <data_dir>/test_pockets.pkl --out-table <data_dir>/test_table.csv
```
#### Processed data
The processed CrossDocked data can be found here:
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

Process the test directory using:
```python
python scripts/prepare_moad.py --test_path <test_directory>  --out-mol-sdf <data_dir>/MOAD_test_mol.sdf --out-pockets-pkl <data_dir>/MOAD_pockets.pkl --out-table <data_dir>/MOAD_table.csv
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
- `num_pharma_atoms`: parameter specifies the number of pharmacophore atoms used for molecule generation from the extracted ones.
- `distance_th sets`: the threshold for determining whether a pharmacophore is satisfied.
An example command is:
```python
python scripts/sample_single_pocket.py --outdir ./outputs --config ./configs/sample/sample_MolDiff.yml --batch_size 32 --pocket_path ./data/example_1h00/processed_pocket_1h00.pkl --sdf_path ./data/example_1h00/ref_points.sdf --use_pharma False --num_pharma_atoms 20 --clash_rate 0.1
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

To evaluate the generated molecules, run the following command:
```python
python scripts/evaluate_all.py --result_root <result_root> --exp_name <exp_name> --from_where generated
```
The parameters are:
- `result_root`: the parent directory of the directory of the sampled molecules (i.e, the same as the `outdir` parameter when running `sample_drug3d.py`).
- `exp_name`: the name (or prefix) of the directory of the molecules (excluding the suffix `_SDF`).
- `from_where`: be one of `generated` of `dataset`.

An example command to calculate metrics for the sampled molecules is:
```python
python scripts/evaluate_all.py --result_root ./outputs --exp_name sample_MolDiff_20230101_000000 --from_where generated
```






