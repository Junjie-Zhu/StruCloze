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

from src.data.dataset import InferenceDataset, BioInferenceDataset
from src.data.transform import FeatureTransform, BioFeatureTransform, convert_atom_name_id
from src.data.cropping import single_chain_choice
from src.data.dataloader import get_inference_dataloader
from src.model.integral import FoldEmbedder
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.pdb_utils import to_pdb, to_mmcif

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
    dataset = BioInferenceDataset(
        path_to_dataset=args.data.path_to_dataset,
        suffix=args.data.suffix,
        transform=BioFeatureTransform(
            recenter_atoms=args.data.recenter_atoms,
            eps=args.data.eps,
            training=False,
            repr=args.data.repr
        ),
    )
    inference_loader = get_inference_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        distributed=DIST_WRAPPER.world_size > 1,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
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
        logging.info(f"Loaded checkpoint from {args.ckpt_dir}")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")

    # sanity check
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for inference_iter, inference_batch in tqdm(enumerate(inference_loader)):
            accession_code = inference_batch["accession_code"]

            chain_num = torch.max(inference_batch["chain_index"]).item() + 1
            if chain_num > 62:
                inference_batch = {k: v.squeeze() for k, v in inference_batch.items() if isinstance(v, torch.Tensor)}

                pred_structure = []
                for truncated_batch in single_chain_choice(inference_batch):
                    truncated_batch = to_device({k: v.unsqueeze(0) for k, v in truncated_batch.items()}, device)
                    init_positions = truncated_batch["ref_structure"].unsqueeze(1)

                    # recenter for each chain
                    init_positions = init_positions - torch.mean(init_positions, dim=-2, keepdim=True)

                    pred_positions = model(
                        initial_positions=init_positions,
                        input_feature_dict=truncated_batch,
                    )

                    pred_structure.append(align_with_cg(pred_positions.squeeze(), truncated_batch['atom_com'].squeeze(), truncated_batch)[0])
                pred_structure = torch.cat(pred_structure, dim=0)

            else:
                inference_batch = to_device(inference_batch, device)
                init_positions = inference_batch["ref_structure"].unsqueeze(1)

                # recenter for each chain
                init_positions = init_positions - torch.mean(init_positions, dim=-2, keepdim=True)

                pred_structure = model(
                    initial_positions=init_positions,
                    input_feature_dict=inference_batch,
                )
                inference_batch["accession_code"] = accession_code

            if chain_num >= 62 or args.save_mmcif:
                to_mmcif(
                    input_feature_dict=inference_batch,
                    atom_positions=pred_structure,
                    output_dir=os.path.join(logging_dir, "samples"),
                )
            else:
                to_pdb(
                    input_feature_dict=inference_batch,
                    atom_positions=pred_structure,
                    output_dir=os.path.join(logging_dir, "samples"),
                )

            torch.cuda.empty_cache()

    # Clean up process group when finished
    if DIST_WRAPPER.world_size > 1:
        dist.destroy_process_group()


def structure_augment(input_feature_dict,
                      n_samples=1):
    token_index = input_feature_dict["token_index"]
    atom_to_token_index = input_feature_dict["atom_to_token_index"]
    B, N_token = token_index.shape
    _, N_atom = atom_to_token_index.shape

    atom_com = input_feature_dict["atom_com"].unsqueeze(1).expand(B, n_samples, N_atom, 3)

    ref_structure = input_feature_dict['ref_positions'].unsqueeze(1).expand(B, n_samples, N_atom, 3) + atom_com
    return ref_structure


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


def align_with_cg(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    feature_dict: dict,
):
    weight = feature_dict["atom_mask"].float().squeeze()
    ca_mask = convert_atom_name_id(feature_dict["ref_atom_name_chars"].squeeze())
    ca_mask = [i == "CA" for i in ca_mask]

    pred_ca = pred_pose[ca_mask, :]
    true_ca = true_pose[ca_mask, :]
    weight = weight[ca_mask]

    weighted_n_atoms = torch.sum(weight, dim=-1, keepdim=True).unsqueeze(-1)
    pred_pose_centroid = (
        torch.sum(pred_ca * weight.unsqueeze(-1), dim=-2, keepdim=True)
        / weighted_n_atoms
    )
    pred_pose_centered = pred_ca - pred_pose_centroid
    true_pose_centroid = (
        torch.sum(true_ca * weight.unsqueeze(-1), dim=-2, keepdim=True)
        / weighted_n_atoms
    )
    true_pose_centered = true_ca - true_pose_centroid
    H_mat = torch.matmul(
        (pred_pose_centered * weight.unsqueeze(-1)).transpose(-2, -1),
        true_pose_centered * weight.unsqueeze(-1),
    )
    u, s, v = torch.svd(H_mat)
    u = u.transpose(-1, -2)

    det = torch.linalg.det(torch.matmul(v, u))

    diagonal = torch.stack(
        [torch.ones_like(det), torch.ones_like(det), det], dim=-1
    )
    rot = torch.matmul(
        torch.diag_embed(diagonal).to(u.device),
        u,
    )
    rot = torch.matmul(v, rot)

    translate = true_pose_centroid - torch.matmul(
        pred_pose_centroid, rot.transpose(-1, -2)
    )

    pred_pose_translated = (
        torch.matmul(pred_pose, rot.transpose(-1, -2)) + true_pose_centroid
    )

    return pred_pose_translated, rot, translate


if __name__ == '__main__':
    main()


