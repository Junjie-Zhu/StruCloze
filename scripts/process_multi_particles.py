import biotite.structure as struc
import numpy as np

import biom_constants as bc


def residue_to_martini(residue):
    martini = residue
    return martini


def residue_to_calv_rna(residue):
    assert residue.res_name[0] in ["A", "C", "G", "U", "N"], f"Invalid residue name: {residue.res_name[0]}"


def read_martini_topology(name):
    top_s = {}
    weight_s = {}
    with open(name) as fp:
        for line in fp:
            if line.startswith("RESI"):
                resName = line.strip().split()[1]
                top_s[resName] = []
                weight_s[resName] = []
            elif line.startswith("BEAD"):
                atmName_s = line.strip().split()[2:]
                weights = []
                for i_atm, atmName in enumerate(atmName_s):
                    if "_" in atmName:
                        atmName, weight = atmName.split("_")
                        atmWeight = bc.WEIGHT_MAPPING[atmName[0]]
                        weight = weight.split("/")
                        weight = float(weight[0]) / float(weight[1]) * atmWeight
                    else:
                        weight = bc.WEIGHT_MAPPING[atmName[0]]
                    weights.append(weight)
                top_s[resName].append(atmName_s)
                weight_s[resName].append(weights)
    return top_s, weight_s


print(read_martini_topology('martini3.top'))