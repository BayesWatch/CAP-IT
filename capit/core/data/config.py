from dataclasses import dataclass


@dataclass
class ImageShape:
    channels: int = 3
    width: int = 224
    height: int = 224


@dataclass
class ModalityConfig:
    image: bool = True
    audio: bool = False
    video: bool = False
    text: bool = True
