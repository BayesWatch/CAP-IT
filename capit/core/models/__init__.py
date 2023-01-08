import copy
from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor

from .base import *
from .baselines import *
from .helpers import *


@dataclass
class ModelAndTransform:
    model: nn.Module
    transform: Any
