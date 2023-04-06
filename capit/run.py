import os
import shutil
import neptune

import wandb
from rich import print
from rich.traceback import install

from capit.core.data.datasets import (
    SplitType,
)
from capit.core.utils.storage import save_json

os.environ[
    "HYDRA_FULL_ERROR"
] = "1"  # Makes sure that stack traces produced by hydra instantiation functions produce
# traceback errors related to the modules they built, rather than generic instantiate related errors that
# are generally useless for debugging

os.environ[
    "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"  # extremely useful when debugging DDP setups

install()  # beautiful and clean tracebacks for debugging


import pathlib
from typing import Callable, List, Optional, Union

import hydra
import torch
from huggingface_hub import (
    Repository,
    create_repo,
    hf_hub_download,
    login,
    snapshot_download,
)
from hydra_zen import instantiate
from capit.boilerplate import Learner
from capit.callbacks import Callback
from capit.config import BaseConfig, collect_config_store
from capit.evaluators import ClassificationEvaluator
from capit.trainers import ClassificationTrainer
from capit.utils import (
    create_hf_model_repo_and_download_maybe,
    get_logger,
    pretty_config,
    set_seed,
)
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

config_store = collect_config_store()

logger = get_logger(name=__name__)


def instantiate_callbacks(callback_dict: dict) -> List[Callback]:
    callbacks = []
    for cb_conf in callback_dict.values():
        callbacks.append(instantiate(cb_conf))

    return callbacks


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: BaseConfig) -> None:
    ckpt_path, repo_url = create_hf_model_repo_and_download_maybe(cfg)

    if ckpt_path is not None:
        logger.info(
            f"ckpt_path: {ckpt_path}, exists: {ckpt_path.exists()}, resume: {cfg.resume}, not resume: {not cfg.resume}"
        )
    else:
        logger.info(
            f"ckpt_path: {ckpt_path}, resume: {cfg.resume}, not resume: {not cfg.resume}"
        )

    logger.info(f"Using checkpoint: {ckpt_path}")

    print(pretty_config(cfg, resolve=True))

    set_seed(seed=cfg.seed)

    if ckpt_path is not None and cfg.resume is True:
        trainer_state = torch.load(pathlib.Path(ckpt_path) / "trainer_state.pt")
        global_step = trainer_state["global_step"]
        neptune_id = (
            trainer_state["neptune_id"]
            if "neptune_id" in trainer_state
            else None
        )
        experiment_tracker = neptune.init_run(
            source_files=["capit/*.py", "kubernetes/*.py"], with_id=neptune_id
        )
    else:
        global_step = 0
        experiment_tracker = neptune.init_run(
            source_files=["capit/*.py", "kubernetes/*.py"]
        )

    wandb.init()
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    experiment_tracker["config"] = config_dict
    experiment_tracker["notes"] = repo_url
    experiment_tracker["init_global_step"] = global_step

    wandb.config.update(config_dict)
    wandb.config.update({"notes": repo_url})
    wandb.config.update({"init_global_step": global_step})

    model: nn.Module = instantiate(cfg.model)

    train_dataset: Dataset = instantiate(
        cfg.dataset,
        set_name=SplitType.TRAIN,
        num_episodes=cfg.total_train_steps,
    )
    val_dataset: Dataset = instantiate(
        cfg.dataset,
        set_name=SplitType.VAL,
        num_episodes=cfg.total_val_steps,
    )
    test_dataset: Dataset = instantiate(
        cfg.dataset,
        set_name=SplitType.TEST,
        num_episodes=cfg.total_test_steps,
    )

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
    )
    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
    )
    test_dataloader = instantiate(
        cfg.dataloader,
        dataset=test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
    )

    if hasattr(model, "is_built") and model.is_built == False:
        model.build(next(iter(train_dataloader)))

    params = model.parameters()

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=params, _partial_=False
    )

    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
        cfg.scheduler,
        optimizer=optimizer,
        t_initial=cfg.learner.train_iters,
        _partial_=False,
    )

    learner: Learner = instantiate(
        cfg.learner,
        model=model,
        trainers=[
            ClassificationTrainer(
                optimizer=optimizer,
                scheduler=scheduler,
                experiment_tracker=experiment_tracker,
            )
        ],
        evaluators=[
            ClassificationEvaluator(experiment_tracker=experiment_tracker)
        ],
        train_dataloaders=[train_dataloader],
        val_dataloaders=[val_dataloader],
        callbacks=instantiate_callbacks(cfg.callbacks),
        resume=ckpt_path,
    )

    if cfg.train:
        learner.train()

    if cfg.test:
        learner.test(test_dataloaders=[test_dataloader])


if __name__ == "__main__":
    run()
