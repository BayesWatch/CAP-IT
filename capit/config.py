import multiprocessing
import os
from dataclasses import MISSING, dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
from accelerate import Accelerator
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, ZenField, builds, hydrated_dataclass, make_config

from capit.core.data.config import ImageShape, ModalityConfig
from capit.core.data.datasets import (
    ChallengeSamplesSourceTypes,
    InstagramImageTextMultiModalDatasePyArrow,
    SplitType,
    dataclass_collate,
)
from capit.core.models.baselines import (
    CLIPImageTextModel,
    CLIPWithPostProcessingImageTextModel,
)
from .boilerplate import Learner
from .callbacks import UploadCheckpointsToHuggingFace
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from .core.data import build_dataset
from .core.models import build_model

CHECKPOINT_DIR = "${hf_repo_dir}"
HF_USERNAME = "${hf_username}"
CODE_DIR = "${code_dir}"
DATASET_DIR = "${data_dir}"
EXPERIMENT_NAME = "${exp_name}"
EXPERIMENTS_ROOT_DIR = "${root_experiment_dir}"
TRAIN_BATCH_SIZE = "${train_batch_size}"
CURRENT_EXPERIMENT_DIR = "${current_experiment_dir}"
TRAIN_ITERS = "${learner.train_iters}"
REPO_PATH = "${repo_path}"
EXP_NAME = "${exp_name}"
SEED = "${seed}"
RESUME = "${resume}"


@hydrated_dataclass(target=DataLoader)
class DataLoaderConfig:
    dataset: Any = None
    batch_size: int = TRAIN_BATCH_SIZE
    persistent_workers: bool = False
    pin_memory: bool = True
    prefetch_factor: int = 2
    num_workers: int = multiprocessing.cpu_count()
    shuffle: bool = True


@dataclass
class DatasetDirectoryConfig:
    train: Optional[str] = None
    val: Optional[str] = None
    test: Optional[str] = None


@dataclass
class DatasetConfig:
    _target_: Any = None
    dataset_dir_config: DatasetDirectoryConfig = DatasetDirectoryConfig()
    set_name: str = "dummy"
    num_samples: int = 100
    using_pre_sampled_split: bool = False
    dataset_size_identifier: str = "base"
    dataset_name: str = "base"
    modality_config: ModalityConfig = ModalityConfig()
    rescan_paths: bool = False
    num_video_frames_per_datapoint: int = 10
    num_audio_frames_per_datapoint: int = 88200
    num_audio_sample_rate: int = 44100
    image_shape: ImageShape = ImageShape(channels=3, width=224, height=224)
    text_context_length: int = 77


wandb_args_config = builds(wandb.init, populate_full_signature=True)

wandb_args_default = wandb_args_config(
    project=os.environ["WANDB_PROJECT"],
    resume="allow",  # allow, True, False, must
    dir=CURRENT_EXPERIMENT_DIR,
    save_code=True,
)


@hydrated_dataclass(target=timedelta)
class TimerConfig:
    seconds: int = 60
    # minutes: int = 60


HFModelUploadConfig = builds(
    UploadCheckpointsToHuggingFace, populate_full_signature=True
)

hf_upload = HFModelUploadConfig(repo_name=EXPERIMENT_NAME, repo_owner=HF_USERNAME)

adamw_optimizer_config = builds(
    torch.optim.AdamW,
    populate_full_signature=True,
    zen_partial=True,
)


cosine_learning_rate_scheduler_config = builds(
    CosineLRScheduler,
    populate_full_signature=True,
    zen_partial=True,
)

accelerator_config = builds(Accelerator, populate_full_signature=True)

cosine_learning_rate_scheduler_config = cosine_learning_rate_scheduler_config()

baseline_model_config = CLIPImageTextModel.build_config(populate_full_signature=True)
fine_tuning_model_config = CLIPWithPostProcessingImageTextModel.build_config(
    populate_full_signature=True
)

dataset_config = InstagramImageTextMultiModalDatasePyArrow.build_config(
    populate_full_signature=True
)

dataloader_config = builds(
    DataLoader,
    dataset=None,
    collate_fn=dataclass_collate,
    populate_full_signature=True,
)

learner_config = builds(Learner, populate_full_signature=True)

learner_config = learner_config(
    model=None,
    experiment_name=EXPERIMENT_NAME,
    experiment_dir=CHECKPOINT_DIR,
    resume=RESUME,
    evaluate_every_n_steps=2500,
    checkpoint_after_validation=True,
    checkpoint_every_n_steps=500,
    train_iters=1000000,
)

default_callbacks = dict(hf_uploader=hf_upload)


@dataclass
class BaseConfig:

    # Must be passed at command line -- neccesary arguments

    exp_name: str = MISSING

    # Defaults for these are provided in the collect_config_store method,
    # but will be often overridden at command line

    model: Any = MISSING
    dataset: Any = MISSING
    dataloader: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    learner: Any = MISSING
    callbacks: Any = MISSING

    wandb_args: Any = MISSING

    hf_username: str = (
        os.environ["HF_USERNAME"] if "HF_USERNAME" in os.environ else MISSING
    )

    seed: int = 42

    freeze_backbone: bool = False
    resume: bool = False
    resume_from_checkpoint: Optional[int] = None
    print_config: bool = False
    train_batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = multiprocessing.cpu_count()
    train: bool = True
    test: bool = False
    download_latest: bool = True
    download_checkpoint_with_name: Optional[str] = None

    root_experiment_dir: str = (
        os.environ["EXPERIMENTS_DIR"]
        if "EXPERIMENTS_DIR" in os.environ
        else "/experiments"
    )

    data_dir: str = (
        os.environ["DATASET_DIR"] if "DATASET_DIR" in os.environ else "/data"
    )

    current_experiment_dir: str = "${root_experiment_dir}/${exp_name}"
    repo_path: str = "${hf_username}/${exp_name}"
    hf_repo_dir: str = "${current_experiment_dir}/repo"
    code_dir: str = "${hydra:runtime.cwd}"


# Using hydra might look a bit more verbose but it saves having to manually define
# future args, and makes it a lot easier to add whatever we need from the command line


def collect_config_store():

    config_store = ConfigStore.instance()
    ###################################################################################
    baseline_config = baseline_model_config(
        model_name_or_path="openai/clip-vit-large-patch14",
        pretrained=True,
    )

    baseline_with_post_processing_config = fine_tuning_model_config(
        model_name_or_path="openai/clip-vit-large-patch14",
        pretrained=True,
        backbone_fine_tunable=False,
    )

    instait_config = dataset_config(
        dataset_dir=DATASET_DIR,
        set_name=SplitType.TRAIN,
        top_k_percent=25,
        reset_cache=False,
        num_episodes=100,
        max_num_collection_images_per_episode=0,
        max_num_query_images_per_episode=50,
        challenge_image_source=ChallengeSamplesSourceTypes.WITHIN_USER,
    )

    ###################################################################################

    config_store.store(
        group="model",
        name="clip-baseline",
        node=baseline_config,
    )

    config_store.store(
        group="model",
        name="clip-with-post-processing-baseline",
        node=baseline_with_post_processing_config,
    )

    config_store.store(
        group="dataset",
        name="instait",
        node=instait_config,
    )

    config_store.store(
        group="dataloader",
        name="default",
        node=dataloader_config(
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
            shuffle=True,
        ),
    )

    config_store.store(
        group="optimizer",
        name="adamw",
        node=adamw_optimizer_config(lr=0.00002, weight_decay=0.00),
    )

    config_store.store(
        group="scheduler",
        name="cosine-annealing",
        node=cosine_learning_rate_scheduler_config,
    )

    ###################################################################################
    config_store.store(
        group="learner",
        name="default",
        node=learner_config,
    )

    config_store.store(group="callbacks", name="default", node=default_callbacks)

    config_store.store(group="wandb_args", name="default", node=wandb_args_default)

    config_store.store(
        group="hydra",
        name="default",
        node=dict(
            job_logging=dict(
                version=1,
                formatters=dict(
                    simple=dict(
                        level="INFO",
                        format="%(message)s",
                        datefmt="[%X]",
                    )
                ),
                handlers=dict(
                    rich={
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                ),
                root={"handlers": ["rich"], "level": "INFO"},
                disable_existing_loggers=False,
            ),
            hydra_logging=dict(
                version=1,
                formatters=dict(
                    simple=dict(
                        level="INFO",
                        format="%(message)s",
                        datefmt="[%X]",
                    )
                ),
                handlers={
                    "rich": {
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                },
                root={"handlers": ["rich"], "level": "INFO"},
                disable_existing_loggers=False,
            ),
            run={"dir": "${current_experiment_dir}/hydra-run/${now:%Y-%m-%d_%H-%M-%S}"},
            sweep={
                "dir": "${current_experiment_dir}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
                "subdir": "${hydra.job.num}",
            },
        ),
    )

    zen_config = []

    for value in BaseConfig.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            dict(learner="default"),
            dict(optimizer="adamw"),
            dict(scheduler="cosine-annealing"),
            dict(model="clip-baseline"),
            dict(dataset="instait"),
            dict(dataloader="default"),
            dict(hydra="default"),
            dict(wandb_args="default"),
            dict(callbacks="default"),
        ],
    )
    # Config
    config_store.store(name="config", node=config)

    return config_store
