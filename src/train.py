import logging
import os

import rootutils
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import TrainingDataset, FeatureTransform
from src.data.dataloader import get_training_dataloader
from src.model.integral import FoldEmbedder
from src.model.loss import AllLosses
from src.model.optimizer import get_optimizer, get_lr_scheduler
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config_train")
def main(args: DictConfig):
    # update logging directory with current time
    if not os.path.isdir(args.logging_dir):
        os.makedirs(args.logging_dir)
    logging_dir = os.path.join(args.logging_dir, f"TRAIN_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(logging_dir)
    os.makedirs(os.path.join(logging_dir, "checkpoints"))  # for saving checkpoints

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
    dataset = TrainingDataset(
        path_to_dataset=args.data.path_to_dataset,
        transform=FeatureTransform(
            truncate_size=args.data.truncate_size,
            recenter_atoms=args.data.recenter_atoms,
            eps=args.data.eps,
            ccd_info=args.data.path_to_ccd_info
        ),
    )
    train_loader, val_loader = get_training_dataloader(
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
    optimizer = get_optimizer(
        model,
        lr=args.optimizer.lr,
        weight_decay=args.optimizer.weight_decay,
        betas=(args.optimizer.beta1, args.optimizer.beta2),
        use_adamw=args.optimizer.use_adamw
    )
    scheduler = get_lr_scheduler(
        optimizer,
        lr_scheduler=args.optimizer.lr_scheduler,  # by default, use af3 scheduler
        lr=args.optimizer.lr,
        max_steps=args.epochs * len(train_loader) + 100,
        warmup_steps=args.optimizer.warmup_steps,
        decay_every_n_steps=args.optimizer.decay_every_n_steps,
        decay_factor=args.optimizer.decay_factor,
    )

    start_epoch = 1
    if args.resume.ckpt_dir is not None:
        # load checkpoint (.pth)
        checkpoint = torch.load(args.ckpt_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not args.resume.load_model_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Loaded checkpoint from {args.ckpt_dir}")

    # instantiate loss
    loss_fn = AllLosses(
        weight_mse=args.loss.weight_mse,
        eps=args.loss.eps,
        reduction=args.loss.reduction
    )

    # Initialize variables for storing loss data
    step_losses, epoch_losses, val_losses = [], [], []
    with open(f"{logging_dir}/loss.csv", "w") as f:
        f.write("Epoch,Loss,Val Loss\n")

    epoch_progress = tqdm(
        total=args.epochs,
        leave=False,
        position=0,
        ncols=100,  # Adjust width of the progress bar
    )
    # Main train/eval loop
    for crt_epoch in range(start_epoch, args.epochs + 1):
        epoch_loss, epoch_val_loss = 0, 0
        model.train()

        # Training loop with dynamic progress bar
        with tqdm(
                enumerate(train_loader),
                desc=f"Epoch {crt_epoch}/{args.epochs} ",
                total=len(train_loader),
                leave=True,
                position=1,
                ncols=100,  # Adjust width of the progress bar
        ) as pbar:
            for crt_step, input_feature_dict in pbar:
                input_feature_dict = to_device(input_feature_dict, device)
                init_positions = input_feature_dict["ref_structure"]
                input_feature_dict.pop("ref_structure")

                pred_positions = model(
                    initial_positions=init_positions,
                    input_feature_dict=input_feature_dict,
                )

                loss = loss_fn(pred_positions,
                    input_feature_dict['atom_positions'],
                    single_mask=input_feature_dict['atom_mask'],
                    pair_mask=input_feature_dict['lddt_mask'],
                    lddt_enabled=args.loss.lddt_enabled,
                    bond_enabled=args.loss.bond_enabled
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                step_loss = loss.item()
                epoch_loss += step_loss
                step_losses.append(step_loss)

                # Update the progress bar dynamically
                pbar.set_postfix(step_loss=f"{step_loss:.3f}")

            # Calculate average epoch loss
            epoch_loss /= (crt_step + 1)
            epoch_losses.append(epoch_loss)

        # Validation loop with dynamic progress bar
        model.eval()
        with tqdm(
                enumerate(val_loader),
                desc="Validation",
                total=len(val_loader),
                leave=True,
                position=1,
                ncols=100,  # Adjust width of the progress bar
        ) as val_pbar:
            for crt_val_step, val_feature_dict in val_pbar:
                val_feature_dict = to_device(val_feature_dict, device)
                init_positions = val_feature_dict["ref_structure"]
                val_feature_dict.pop("ref_structure")

                pred_positions = model(
                    initial_positions=init_positions,
                    input_feature_dict=val_feature_dict,
                )
                val_loss = loss_fn(pred_positions,
                    val_feature_dict['atom_positions'],
                    single_mask=val_feature_dict['atom_mask'],
                    pair_mask=input_feature_dict['lddt_mask'],
                    lddt_enabled=args.loss.lddt_enabled,
                    bond_enabled=args.loss.bond_enabled
                )

                step_val_loss = val_loss.item()
                epoch_val_loss += step_val_loss
                val_losses.append(step_val_loss)

                # Update the validation progress bar dynamically
                val_pbar.set_postfix(val_loss=f"{step_val_loss:.3f}")

            # Calculate average validation loss
            epoch_val_loss /= (crt_val_step + 1)

        # update epoch progress bar
        epoch_progress.set_postfix(
            loss=f"{epoch_loss:.3f}", val_loss=f"{epoch_val_loss:.3f}"
        )
        epoch_progress.update()

        # Optionally save the loss data to a file after each epoch
        with open(f"{logging_dir}/loss.csv", "a") as f:
            f.write(f"{crt_epoch},{epoch_loss},{epoch_val_loss}\n")

        # Implement checkpoint saving
        if crt_epoch % args.checkpoint_interval == 0 or crt_epoch == args.epochs:
            checkpoint_path = os.path.join(logging_dir, f"checkpoints/epoch_{crt_epoch}.pth")
            torch.save({
                'epoch': crt_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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


if __name__ == '__main__':
    main()