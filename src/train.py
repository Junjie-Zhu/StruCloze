import argparse

import torch

from src.data.dataset import ProteinDataset
from src.model.integral import FoldEmbedder


def main(args):
    # instantiate dataset
    dataset = ProteinDataset()

    # instantiate model
    model = FoldEmbedder()

    # check if args.strategy is ddp
    if args.strategy == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(model)
