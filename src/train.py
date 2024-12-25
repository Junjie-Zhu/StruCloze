import argparse
import logging
import os
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.data.dataset import ProteinDataset, ProteinTransform
from src.model.integral import FoldEmbedder
from src.model.loss import ScoreMatchingLoss
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.model_utils import (
    get_optimizer,
    get_lr_scheduler,
    get_dataloader)


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
    scheduler = get_lr_scheduler(args.scheduler, optimizer)

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

    # instantiate loss
    loss = ScoreMatchingLoss(args.loss)

    # Initialize variables for storing loss data
    step_losses, epoch_losses, val_losses = [], [], []
    if args.logging_dir is not None:
        with open(f"{args.logging_dir}/loss.csv", "w") as f:
            f.write("Epoch,Loss,Val Loss\n")

    # Main train/eval loop
    for crt_epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_val_loss = 0, 0
        model.train()

        # Training loop with dynamic progress bar
        with tqdm(
                enumerate(train_loader),
                desc=f"Epoch {crt_epoch}/{args.epochs} "
                     f"| Step Loss N/A "
                     f"| Val Loss N/A",
                total=len(train_loader),
                ncols=100,  # Adjust width of the progress bar
        ) as pbar:
            for crt_step, batch in pbar:
                batch = to_device(batch, device)
                out_batch = model(batch)
                loss = loss(out_batch, batch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                step_loss = loss.item()
                epoch_loss += step_loss
                step_losses.append(step_loss)

                # Update the progress bar dynamically
                pbar.set_postfix(step_loss=f"{step_loss:.3f}", epoch_loss=f"{epoch_loss / (crt_step + 1):.3f}")

            # Calculate average epoch loss
            epoch_loss /= (crt_step + 1)
            epoch_losses.append(epoch_loss)

        # Validation loop with dynamic progress bar
        model.eval()
        with tqdm.tqdm(
                enumerate(test_loader),
                desc=f"Val {crt_epoch} | Epoch Loss {epoch_loss:.3f}",
                total=len(test_loader),
                ncols=100,  # Adjust width of the progress bar
        ) as val_pbar:
            for crt_val_step, val_batch in val_pbar:
                val_batch = to_device(val_batch, device)
                with torch.no_grad():
                    out_batch = model(val_batch)
                    val_loss = loss(out_batch, val_batch)

                step_val_loss = val_loss.item()
                epoch_val_loss += step_val_loss
                val_losses.append(step_val_loss)

                # Update the validation progress bar dynamically
                val_pbar.set_postfix(val_loss=f"{step_val_loss:.3f}",
                                     epoch_val_loss=f"{epoch_val_loss / (crt_val_step + 1):.3f}")

            # Calculate average validation loss
            epoch_val_loss /= (crt_val_step + 1)

        # Optionally save the loss data to a file after each epoch
        with open(f"{args.logging_dir}/loss.csv", "a") as f:
            f.write(f"{crt_epoch},{epoch_loss},{epoch_val_loss}\n")

        # Implement checkpoint saving
        if crt_epoch % args.checkpoint_interval == 0 or crt_epoch == args.epochs:
            checkpoint_path = f"checkpoint_epoch_{crt_epoch}.pth"
            torch.save({
                'epoch': crt_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)


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
