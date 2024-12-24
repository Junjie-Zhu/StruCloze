import argparse
import logging
import os
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataset import ProteinDataset, ProteinTransform
from src.model.integral import FoldEmbedder
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.model_utils import get_optimizer, get_dataloader


def main(args):
    # check environment
    use_cuda = torch.cuda.device_count() > 0
    if use_cuda:
        device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        logging.info(
            f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
        )
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if DIST_WRAPPER.world_size > 1:
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )
    # All ddp process got the same seed
    seed_everything(
        seed=args.seed,
        deterministic=args.deterministic,
    )

    # instantiate model
    model = FoldEmbedder(
        encoder_config=args.model.encoder_config,
        latent_config=args.model.latent_config,
        decoder_config=args.model.decoder_config,
        self_conditioning=args.model.self_conditioning,  # not implemented
    ).to(device)
    if DIST_WRAPPER.world_size > 1:
        logging.info("Using DDP")
        model = DDP(
            model,
            device_ids=[DIST_WRAPPER.local_rank],
            output_device=DIST_WRAPPER.local_rank,
            static_graph=True,
        )
    optimizer = get_optimizer(args.optimizer, model)

    # instantiate dataset
    dataset = ProteinDataset(
        path_to_dataset=args.data.path_to_dataset,
        transform=ProteinTransform(
            unit=args.data.unit,
            truncate_length=args.data.truncate_length,
            strip_missing_residues=args.data.strip_missing_residues,
            recenter_and_scale=args.data.recenter_and_scale,
            eps=args.data.eps,
        ),
        training=args.data.training,  # not implemented
    )
    train_loader, test_loader = get_dataloader(
        dataset=dataset,
    )
