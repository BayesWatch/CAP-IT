import copy
from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor

from .base import *
from .baselines import *
from .cap import *
from .helpers import *


@dataclass
class ModelAndTransform:
    model: nn.Module
    transform: Any


def load_from_repo(repo_path: str, model_name: str, cache_path: str):
    checkpoint_path = hf_hub_download(
        repo_id=repo_path,
        cache_dir=pathlib.Path(cache_path),
        resume_download=True,
        subfolder="checkpoints",
        filename=model_name,
        repo_type="model",
    )
    state = torch.load(checkpoint_path)
    print(list(state.keys()))
    # load_state_dict(state["model"])
    return checkpoint_path
