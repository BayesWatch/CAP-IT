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
    global_step: int
    metrics: Dict
    phase_name: str


class ClassificationEvaluator(Evaluator):
    def __init__(
        self, experiment_tracker: wandb.wandb_sdk.wandb_run.Run = None
    ):
        super().__init__()
        self.state_dict = {}
        self.experiment_tracker = experiment_tracker

    @torch.inference_mode()
    def validation_step(
        self,
        model,
        batch,
        global_step,
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
                self.state_dict.setdefault(key, []).append(value.detach().cpu())

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics={"accuracy": metrics["accuracy"], "loss": loss},
        )

    @torch.inference_mode()
    def test_step(
        self,
        model,
        batch,
        global_step,
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
                self.state_dict.setdefault(key, []).append(value.detach().cpu())

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="test",
            metrics={"accuracy": metrics["accuracy"], "loss": loss},
        )

    @collect_metrics
    def start_validation(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=self.state_dict,
        )

    @collect_metrics
    def start_testing(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        return EvaluatorOutput(
            global_step=global_step,
            phase_name="testing",
            metrics=self.state_dict,
        )

    @collect_metrics
    def end_validation(
        self,
        global_step: int,
    ):
        epoch_metrics = {}
        for key, value in self.state_dict.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()
            logger.info(f"Validation {key}: {epoch_metrics} {len(value)}")

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=epoch_metrics,
        )

    @collect_metrics
    def end_testing(
        self,
        global_step: int,
    ):
        epoch_metrics = {}
        for key, value in self.state_dict.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            global_step=global_step, phase_name="testing", metrics=epoch_metrics
        )
