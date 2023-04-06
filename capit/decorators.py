import functools
import random
import time
from typing import Any, Callable

import torch
from hydra_zen import builds, instantiate
import wandb

from capit.utils import get_logger

logger = get_logger(__name__)


def configurable(func: Callable) -> Callable:
    func.__configurable__ = True

    def build_config(**kwargs):
        return builds(func, **kwargs)

    setattr(func, "build_config", build_config)
    return func


def check_if_configurable(func: Callable, phase_name: str) -> bool:
    return func.__configurable__ if hasattr(func, "__configurable__") else False


def get_next_on_error(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                f"Error occurred at idx {args} with error code {e}, getting the next item instead."
            )
            args = list(args)
            args[1] = args[1] + 1
            args = tuple(args)
            return func(*args, **kwargs)

    return wrapper_collect_metrics


def collect_metrics(func: Callable) -> Callable:
    def collect_metrics(
        metrics_dict: dict(),
        phase_name: str,
        experiment_tracker: Any,
        global_step: int,
    ) -> None:
        for metric_key, computed_value in metrics_dict.items():
            if computed_value is not None:
                value = (
                    computed_value.detach().item()
                    if isinstance(computed_value, torch.Tensor)
                    else computed_value
                )
                experiment_tracker[f"{phase_name}/{metric_key}"].append(value)
                wandb.log(
                    {f"{phase_name}/{metric_key}": value}, step=global_step
                )

    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs):
        outputs = func(*args, **kwargs)
        collect_metrics(
            metrics_dict=outputs.metrics,
            phase_name=outputs.phase_name,
            experiment_tracker=args[0].experiment_tracker,
            global_step=outputs.global_step,
        )
        return outputs

    return wrapper_collect_metrics


if __name__ == "__main__":

    @configurable
    def build_something(batch_size: int, num_layers: int):
        return batch_size, num_layers

    build_something_config = build_something.build_config(
        populate_full_signature=True
    )
    dummy_config = build_something_config(batch_size=32, num_layers=2)
    print(dummy_config)

    from hydra_zen import builds, instantiate

    def build_something(batch_size: int, num_layers: int):
        return batch_size, num_layers

    dummy_config = builds(build_something, populate_full_signature=True)

    dummy_function_instantiation = instantiate(dummy_config)

    print(dummy_function_instantiation)
