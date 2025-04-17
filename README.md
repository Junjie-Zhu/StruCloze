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

We have uploaded our training and test dataset to Zenodo. Training set contains `52,926` processed entries from PDB (in format `.pkl.gz`)

