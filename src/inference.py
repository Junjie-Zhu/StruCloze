import logging
import os
import warnings

import rootutils
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import InferenceDataset, FeatureTransform
from src.data.dataloader import get_inference_dataloader
from src.model.integral import FoldEmbedder
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.model_utils import centre_random_augmentation
from src.utils.pdb_utils import to_pdb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config_inference")
def main(args: DictConfig):
    logging_dir = os.path.join(args.logging_dir, f"INF_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    if DIST_WRAPPER.rank == 0:
        # update logging directory with current time
        if not os.path.isdir(args.logging_dir):
            os.makedirs(args.logging_dir)
        os.makedirs(logging_dir)
        os.makedirs(os.path.join(logging_dir, "samples"))  # for saving output models

        # save current configuration in logging directory
        with open(f"{logging_dir}/config.yaml", "w") as f:
            OmegaConf.save(args, f)

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
        logging.info(
            f"Using DDP with {DIST_WRAPPER.world_size} processes, rank: {DIST_WRAPPER.rank}"
        )
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )
    # All ddp process got the same seed
    seed_everything(
        seed=args.seed,
        deterministic=args.deterministic,
    )

    # instantiate dataset
    dataset = InferenceDataset(
        path_to_dataset=args.data.path_to_dataset,
        suffix='pkl',
        transform=FeatureTransform(
            recenter_atoms=args.data.recenter_atoms,
            eps=args.data.eps,
            ccd_info=args.data.path_to_ccd_info
        ),
    )
    inference_loader = get_inference_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        prop_train=args.data.prop_train,
        shuffle=args.data.shuffle,
        distributed=DIST_WRAPPER.world_size > 1,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
        seed=args.seed,
    )

    # instantiate model
    model = FoldEmbedder(
        c_atom=args.model.c_atom,
        c_atompair=args.model.c_atompair,
        c_token=args.model.c_token,
        c_s=args.model.c_s,
        c_z=args.model.c_z,
        n_atom_layers=args.model.n_atom_layers,
        n_token_layers=args.model.n_token_layers,
        n_atom_attn_heads=args.model.n_atom_attn_heads,
        n_token_attn_heads=args.model.n_token_attn_heads,
        initialization=args.model.initialization,
    ).to(device)
    if DIST_WRAPPER.world_size > 1:
        logging.info("Using DDP")
        model = DDP(
            model,
            device_ids=[DIST_WRAPPER.local_rank],
            output_device=DIST_WRAPPER.local_rank,
            static_graph=True,
        )

    assert os.path.isfile(args.ckpt_dir), f"Checkpoint file not found: {args.ckpt_dir}"
    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    if DIST_WRAPPER.world_size > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if DIST_WRAPPER.rank == 0:
        logging.info(f"Loaded checkpoint from {args.resume.ckpt_dir}")
        logging.info(model)
        logging.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # sanity check
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for inference_iter, inference_batch in enumerate(inference_loader):
            inference_batch = to_device(inference_batch, device)
            init_positions = centre_random_augmentation(inference_batch["ref_structure"])
            inference_batch.pop("ref_structure")

            pred_positions = model(
                initial_positions=init_positions,
                input_feature_dict=inference_batch,
            )

            to_pdb(
                input_feature_dict=inference_batch,
                atom_positions=pred_positions,
                output_dir=logging_dir,
            )

            torch.cuda.empty_cache()

    # Clean up process group when finished
    if DIST_WRAPPER.world_size > 1:
        dist.destroy_process_group()


def to_device(obj, device):
    """Move tensor or dict of tensors to device"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                to_device(v, device)
            elif isinstance(v, torch.Tensor):
                obj[k] = obj[k].to(device)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        raise Exception(f"type {type(obj)} not supported")
    return obj


if __name__ == '__main__':
    main()