train_config = {
    "epochs": 500,
    "checkpoint_interval": 10,

    "model": {
        "encoder_config": {

        },
        "latent_config": {

        },
        "decoder_config": {

        },
        "self_conditioning": False,
    },

    "optimizer": {

    },
    "scheduler": {

    },

    "data": {
        "path_to_dataset": None,
        "unit": "angstrom",
        "truncate_length": 384,
        "strip_missing_residues": True,
        "recenter_and_scale": True,
        "eps": 1e-8,
        "training": False,
        "dataloader": {

        },
    },

    "loss": {

    },

    "logging_dir": None,
    "seed": 42,
    "deterministic": True,
}