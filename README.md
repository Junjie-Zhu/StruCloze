# Completing Any Missing Structural Elements in Biomolecules with StruCloze
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![biorXiv](https://img.shields.io/badge/biorxiv.2024.05.05.592611-B31B1B)](https://www.biorxiv.org/content/10.1101/2024.05.05.592611v2)

This repository contains code for "**Completing Any Missing Structural Elements in Biomolecules with StruCloze**", which raise a unified framework for back-mapping coarse-grained structures and completing missing residues, nucleotides in existing experimental structures.

![Overview of StruCloze](./assets/toc.png)

### Table of contents

* Installation
* Dataset downloading **[Optional]** 
* Usage
  * Back-mapping coarse-grained structures
  * Completing missing tokens
  * Analyzing predicted structures
* Train
* Some existing bugs
* Contact
* Acknowledgement

------

### Installation

Conda is recommended for setting up dependencies. To quickly set up an environment, run:

```bash
# Clone this repository and set up virtual environment
git clone https://github.com/Junjie-Zhu/StruCloze.git
cd StruCloze

# Create and activate environment
conda env create -f strucloze.yaml
source activate strucloze

# Install strucloze as a package.
pip install -e .
```

### Dataset downloading

We have uploaded our training and test dataset to Zenodo. Training set contains `52,926` processed entries from PDB (in format `.pkl.gz`), in which features are organized as:

```yaml
# atom features
"atom_positions": torch.FloatTensor  # shape: (N_atom, 3)
"atom_to_token_index": torch.LongTensor  # shape: (N_atom, ), recording token index for each atom

# residue features
"aatype": torch.LongTensor  # shape: (N_residue, ), recording residue type as numbers (range from 0 to 30)
"moltype": torch.IntTensor  # shape: (N_residue, ), recording molecule type as numbers (0 for protein, 1 for rna, 2 for dna)
"residue_index": torch.IntTensor  # shape: (N_residue, ), residue index in each chain
"token_index": torch.IntTensor  # shape: (N_residue, ), token index in the whole bioassembly
"chain_index": torch.IntTensor  # shape: (N_residue, )

# CCD reference features
"ref_positions": torch.FloatTensor  # shape: (N_residue, 3), reference coordinates
"ref_element": torch.IntTensor  # shape: (N_residue, 32), reference element number converted into one-hot
"ref_atom_name_chars": torch.IntTensor  # shape: (N_residue, 4, 64), reference atom name converted into one-hot 

# CG representations
"atom_ca": torch.FloatTensor  # shape: (N_residue, 3), coordinates of CA atoms
"atom_com": torch.FloatTensor  # shape: (N_residue, 3), coordinates of center of mass
"ref_ca": torch.FloatTensor  # shape: (N_residue, 3), CA atoms in CCD
"ref_com": torch.FloatTensor  # shape: (N_residue, 3), center of mass in CCD
"calv_positions": torch.FloatTensor  # shape: (N_residue, 3), CCD reference positions aligned to atom positions
```

Please note that these data are saved as numpy arrays and are processed into tensors in `src/data/transform.py` in iterating dataset.
By default, `calv_positions` are aligned with Martini for proteins and CALVADOS RNA for nucleic acids.

### Usage

#### Back-mapping coarse-grained structures

Extracting CG models from all-atom structures can be done with `scripts/get_cg_repr.py`:

```bash
python scripts/get_cg_repr.py \
  -i /path/to/structure/ \
  -o /path/to/output/cg/
```

Currently CG models are extracted as numpy arrays, we will update pdb format output in the future. 

To backmap CG models with StruCloze, you may refer to the following command:

```bash
python src/inference.py \
  ckpt_dir=/path/to/checkpoint.ckpt \
  data.path_to_dataset=/path/to/cg/dataset/ 
```

If `data.path_to_dataset` is set as a directory, it will automatically search for all `.pdb` files in the directory and back-map them. Metadata file is also supported by setting `data.path_to_dataset` as a file path (`.csv`).

#### Predicting missing residues

Predicting missing residues use exactly the same model, but with a finetuned checkpoint. Usage is the same as having described above.

#### Analyzing predicted structures

We provided structure analysis scripts in `scripts/`. For structures predicted by StruCloze, you may use `scripts/analyze_pdb_simple.py`:
    
```bash
python scripts/analyze_pdb_simple.py \
  -i /path/to/predicted/structure/ \
  -o /path/to/output/metrics/ \
  -r /path/to/reference/structure/feature/
  -p /path/to/metadata.csv  # optional
```

Here the pre-extracted structure features in `.pkl.gz` format are used. If structure features are not available, you may use `scripts/analyze_pdb.py` in which reference structures in `mmcif` or `pdb` format is required instead of features.

There may still occur some errors due to missing residues or weird indexing in reference structures in current scripts, but it handles most siuations in our analysis.

### Train

With our provided dataset (or your own dataset), you can train StruCloze with the following command:

```bash
python src/train.py \
  epoch=200 \
  resume.ckpt_dir=/path/to/your/checkpoint.ckpt \
  data.path_to_dataset=/path/to/your/dataset.csv  # a metadata file is recommended
```

You may alse revise arguments in `configs/config_train.yaml`. For DDP, `batch_size` is set to number of samples on a single card, we use `torchrun` for DDP:

```bash
torchrun --nproc_per_node=8 src/train.py 
```

### Some existing bugs

Currently there are some known bugs in code listed here:

*  The `OXT` atom at the C-terminus is neglected in the prediction for convenience (As all other residues do not contain this atom, we remove it from CCD dictionary).
*  Chain IDs in predicted structures starts from `a` but not `A`.

Any suggestions or bug reports are welcome. Please open an issue in the GitHub repository.

### Contact
If you have any questions, please contact me via email: shiroyuki@sjtu.edu.cn
Or open an issue in the GitHub repository.

### Acknowledgement

A lot thanks to <a href="https://github.com/Immortals-33">Zhuoqi Zheng</a> who helped in idea formation and manuscript presenting, and <a href="">Jinyu Yu</a> who has 