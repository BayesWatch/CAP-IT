from ast import Dict
from dataclasses import dataclass
from typing import Any, Iterator, List, Tuple

import torch
import torch.nn.functional as F
import wandb
from attr import field
from hydra_zen import instantiate
from torch.utils.data import DataLoader
import accelerate

from .decorators import collect_metrics
from .utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    return (
        {
            key: value.shape if isinstance(value, torch.Tensor) else len(value)
            for key, value in x.items()
        }
        if isinstance(x, dict)
        else get_dict_shapes(x.__dict__)
    )


class Evaluator(object):
    def __init__(self):
        pass


@dataclass
class EvaluatorOutput:
    step_idx: int
    metrics: Dict
    phase_name: str


class ClassificationEvaluator(Evaluator):
    def __init__(
        self, experiment_tracker: wandb.wandb_sdk.wandb_run.Run = None
    ):
        super().__init__()
        self.epoch_metrics = {}
        self.experiment_tracker = experiment_tracker

    def validation_step(
        self,
        model,
        batch,
        batch_idx,
        step_idx,
        epoch_idx,
        accelerator: accelerate.Accelerator = None,
    ):
        with torch.no_grad():
            model.eval()
            opt_loss, output_dict = model.forward(
                batch, accelerator=accelerator, step=True
            )
            metrics = output_dict["metrics"]
            loss = opt_loss.detach()

            for key, value in metrics.items():
                self.epoch_metrics.setdefault(key, []).append(
                    value.detach().cpu()
                )

        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="validation",
            metrics={
                "accuracy": metrics["accuracy"],
                "accuracy_top_5": metrics["accuracy_top_5"],
                "loss": loss,
            },
        )

    def test_step(
        self,
        model,
        batch,
        batch_idx,
        step_idx,
        accelerator: accelerate.Accelerator = None,
    ):
        with torch.no_grad():
            model.eval()
            opt_loss, output_dict = model.forward(
                batch, accelerator=accelerator, step=True
            )
            metrics = output_dict["metrics"]
            loss = opt_loss.detach()

            for key, value in metrics.items():
                self.epoch_metrics.setdefault(key, []).append(
                    value.detach().cpu()
                )

        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="test",
            metrics={
                "accuracy": metrics["accuracy"],
                "accuracy_top_5": metrics["accuracy_top_5"],
                "loss": loss,
            },
        )

    @collect_metrics
    def start_validation(
        self,
        epoch_idx: int,
        step_idx: int,
        val_dataloaders: List[DataLoader] = None,
    ):
        self.epoch_metrics = {}
        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="validation",
            metrics=self.epoch_metrics,
        )

    @collect_metrics
    def start_testing(
        self,
        epoch_idx: int,
        step_idx: int,
        test_dataloaders: List[DataLoader] = None,
    ):
        self.epoch_metrics = {}
        return EvaluatorOutput(
            step_idx=step_idx, phase_name="testing", metrics=self.epoch_metrics
        )

    @collect_metrics
    def end_validation(
        self,
        epoch_idx: int,
        step_idx: int,
        val_dataloaders: List[DataLoader] = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()
            logger.info(f"Validation {key}: {epoch_metrics} {len(value)}")

        return EvaluatorOutput(
            step_idx=step_idx, phase_name="validation", metrics=epoch_metrics
        )

    @collect_metrics
    def end_testing(
        self,
        epoch_idx: int,
        step_idx: int,
        test_dataloaders: List[DataLoader] = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            step_idx=step_idx, phase_name="testing", metrics=epoch_metrics
        )
