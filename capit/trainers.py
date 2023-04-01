from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from hydra_zen import instantiate
from capit.callbacks import Interval
from torch.utils.data import DataLoader

from .decorators import collect_metrics
from .utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


class Trainer(object):
    def __init__(self):
        pass


@dataclass
class TrainerOutput:
    opt_loss: torch.Tensor
    global_step: int
    metrics: Dict[str, Any]
    phase_name: str


class ClassificationTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = Interval.STEP,
        experiment_tracker: wandb.wandb_sdk.wandb_run.Run = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.state_dict = {}

        if self.scheduler is not None:
            assert scheduler_interval in {"step", "epoch"}
            self.scheduler_interval = scheduler_interval

    def get_optimizer(self):
        return self.optimizer

    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        model.train()
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(
            enabled=True, dtype=torch.bfloat16
        ) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            opt_loss, output_dict = model.forward(
                batch, accelerator=accelerator, step=True
            )

        loss = opt_loss.detach()
        accelerator.backward(loss=opt_loss)

        self.optimizer.step()

        if self.scheduler is not None and self.scheduler_interval == "step":
            self.scheduler.step(epoch=global_step)

        metrics = output_dict["metrics"]
        for key, value in metrics.items():
            self.state_dict.setdefault(key, []).append(value.detach().cpu())

        return TrainerOutput(
            phase_name="training",
            opt_loss=opt_loss,
            global_step=global_step,
            metrics={
                "accuracy": metrics["accuracy"],
                "loss": loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            },
        )

    @collect_metrics
    def start_training(self, global_step: int):
        self.state_dict = {}
        return TrainerOutput(
            opt_loss=None, global_step=global_step, metrics={}, phase_name="training"
        )

    @collect_metrics
    def end_training(self, global_step):
        epoch_metrics = {}
        for key, value in self.state_dict.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return TrainerOutput(
            opt_loss=None,
            global_step=global_step,
            metrics=epoch_metrics,
            phase_name="training",
        )
