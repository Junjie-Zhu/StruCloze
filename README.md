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

#### Completing missing tokens

### Analyzing predicted structures

### Some existing bugs

Currently there are some known bugs in code listed here:

*  The `OXT` atom at the C-terminus is neglected in the prediction for convenience (As all other residues do not contain this atom, we remove it from CCD dictionary).
*  Chain IDs in predicted structures starts from `a` but not `A`.

Any suggestions or bug reports are welcome. Please open an issue in the GitHub repository.

### Contact
If you have any questions, please contact me via email: shiroyuki@sjtu.edu.cn
Or open an issue in the GitHub repository.

### Acknowledgement

