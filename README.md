# Completing Any Missing Structural Elements in Biomolecules with StruCloze

This repository contains code for "**Completing Any Missing Structural Elements in Biomolecules with StruCloze**", which raise a unified framework for back-mapping coarse-grained structures and completing missing residues, nucleotides in existing experimental structures.

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

