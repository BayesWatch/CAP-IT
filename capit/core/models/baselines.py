from dataclasses import dataclass
import pathlib
from typing import Any, Iterator, List, Optional, Tuple, Union
from unittest.util import strclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torchtyping import TensorType
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput, contrastive_loss
from capit.core.data.datasets import ImageTextRetrievalInput

from capit.core.utils import get_logger
from capit.decorators import configurable
import accelerate
from huggingface_hub import (
    Repository,
    create_repo,
    hf_hub_download,
    login,
    snapshot_download,
)

log = get_logger(__name__)


@dataclass
class CLIPModelOutput:
    logits_per_image: torch.Tensor
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    loss: Optional[torch.Tensor] = None


def top5_accuracy(predictions):
    """
    Computes the top 5 accuracy for a set of predictions and labels.
    """

    return (predictions.argsort(descending=True) == 0).float().sum()


@configurable
class CLIPImageTextModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name_or_path)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name_or_path
        )

        self.pretrained = pretrained

        if not pretrained:
            self.model.init_weights()

        self.model.train()

        self.image_shape = [
            3,
            self.processor.feature_extractor.size,
            self.processor.feature_extractor.size,
        ]
        self.is_built = False

    def build(
        self,
        batch: ImageTextRetrievalInput,
        accelerator: Optional[accelerate.Accelerator] = None,
    ):
        log.info(f"Building {self.__class__.__name__} module")

        image = batch.target_image[0]
        challenge_images = batch.challenge_images[0]
        image = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch.target_text[0]

        image = self.preprocess_image(image)
        text = self.preprocess_text(text)

        if len(text.shape) == 1:
            text = text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=text,
            pixel_values=image,
            output_hidden_states=True,
            return_loss=False,
        )

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]
        text_hidden_token = clip_output.text_model_output.hidden_states[-1]

        self.is_built = True
        log.info(
            f"Built {self.__class__.__name__} \
                image output: {image_hidden_token.shape}, text output: {text_hidden_token.shape}"
        )

    def preprocess_image(
        self, image: TensorType["batch_size", "channel", "height", "width"]
    ):
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image.unbind(0)
        image = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        image = image.to(self.model.device)

        if len(image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                "method forward_image must be 4, instead it is "
                f"{len(image.shape)}, for shape {image.shape}"
            )
        return image

    def preprocess_text(self, text: List[str]) -> torch.Tensor:
        text = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        text = text.to(self.model.device)
        text = text.to(torch.int32)
        return text

    def forward_image(
        self, image: TensorType["batch_size", "channel", "height", "width"]
    ) -> torch.Tensor:
        image = self.preprocess_image(image)
        clip_output = self.model.forward(pixel_values=image)

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        return image_hidden_token

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:
        text = self.preprocess_text(text)
        if len(text.shape) == 1:
            text = text.unsqueeze(0)
        clip_output = self.model.forward(input_ids=text)

        text_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        return text_hidden_token

    def forward(
        self,
        batch: ImageTextRetrievalInput,
        step: bool = False,
        accelerator: Optional[accelerate.Accelerator] = None,
        **kwargs,
    ) -> CLIPOutput:
        image = batch.target_image[0]
        challenge_images = batch.challenge_images[0]
        images = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch.target_text[0]

        challenge_images = self.preprocess_image(images)
        prompt_text = self.preprocess_text(text)

        if len(prompt_text.shape) == 1:
            prompt_text = prompt_text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=prompt_text,
            pixel_values=challenge_images,
            output_hidden_states=True,
            return_loss=False,
        )

        image_output = clip_output.image_embeds
        text_output = clip_output.text_embeds

        if accelerator is not None:
            image_output = accelerator.gather(image_output)
            text_output = accelerator.gather(text_output)

        similarity = (
            torch.matmul(text_output, image_output.t()) * self.model.logit_scale
        )

        loss = contrastive_loss(similarity)

        if step:
            return self.compute_loss_and_accuracy(
                CLIPModelOutput(
                    logits_per_image=similarity,
                    image_embeds=image_output,
                    text_embeds=text_output,
                    loss=loss,
                )
            )
        return CLIPModelOutput(
            logits_per_image=similarity,
            image_embeds=image_output,
            text_embeds=text_output,
            loss=loss,
        )

    def predict_individual(
        self,
        image: TensorType["batch_size", "channel", "height", "width"],
        text: List[str],
    ) -> CLIPModelOutput:
        image_embeds = self.forward_image(image)
        text_embeds = self.forward_text(text)

        image_embeds = image_embeds / image_embeds.norm(
            p=2, dim=-1, keepdim=True
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits

        logit_scale = self.model.logit_scale.exp()
        return torch.sum(text_embeds * image_embeds, dim=1) * logit_scale

    def compute_loss_and_accuracy(
        self,
        clip_output: CLIPModelOutput,
        accelerator: accelerate.Accelerator = None,
    ):
        accuracy = (
            (clip_output.logits_per_image.argmax(dim=-1) == 0).float().mean()
        )
        accuracy_top_5 = top5_accuracy(clip_output.logits_per_image)
        output_dict = clip_output.__dict__
        output_dict["metrics"] = {
            "accuracy": accuracy,
            "accuracy_top_5": accuracy_top_5,
            "loss": clip_output.loss,
        }

        return output_dict["loss"], output_dict

    def load_from_repo(self, repo_path: str, model_name: str, cache_path: str):
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
        self.load_state_dict(state["model"])
        return checkpoint_path


@configurable
class CLIPWithPostProcessingImageTextModel(CLIPImageTextModel):
    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        backbone_fine_tunable: bool = True,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path, pretrained=pretrained
        )
        self.fine_tunable = backbone_fine_tunable

        if not pretrained:
            self.model.init_weights()

        self.model.train()

        self.image_shape = [
            3,
            self.processor.feature_extractor.size,
            self.processor.feature_extractor.size,
        ]
        self.is_built = False
        self.post_processing_module = nn.ModuleDict()

    def parameters(self):
        if self.fine_tunable:
            return list(self.model.parameters()) + list(
                self.post_processing_module.parameters()
            )
        else:
            return self.post_processing_module.parameters()

    def named_parameters(
        self, recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        if self.fine_tunable:
            return list(self.model.named_parameters(recurse=recurse)) + list(
                self.post_processing_module.named_parameters(recurse=recurse)
            )

        else:
            return self.post_processing_module.named_parameters(recurse=recurse)

    def build_post_processing_module(self, name: str, x: torch.Tensor):
        transformer_encoder = nn.TransformerEncoderLayer(
            d_model=x.shape[2], nhead=8, dim_feedforward=2048
        )
        encoder_norm = nn.LayerNorm(x.shape[2])
        self.post_processing_module[
            f"{name}_transformer"
        ] = nn.TransformerEncoder(
            encoder_layer=transformer_encoder, num_layers=1, norm=encoder_norm
        )
        x = self.post_processing_module[f"{name}_transformer"](x)
        x = x.mean(dim=1)
        self.post_processing_module[f"{name}_output"] = nn.Linear(
            x.shape[1], 512
        )
        x = self.post_processing_module[f"{name}_output"](x)
        return x

    def apply_post_processing(self, x: torch.Tensor, name: str):
        x = self.post_processing_module[f"{name}_transformer"](x)
        x = x.mean(dim=1)
        x = self.post_processing_module[f"{name}_output"](x)
        return x

    def build(self, batch: ImageTextRetrievalInput, **kwargs):
        log.info(f"Building model {self.__class__.__name__}")
        image = batch.target_image[0]
        challenge_images = batch.challenge_images[0]
        image = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch.target_text[0]

        image = self.preprocess_image(image)
        text = self.preprocess_text(text)

        if len(text.shape) == 1:
            text = text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=text,
            pixel_values=image,
            output_hidden_states=True,
            return_loss=False,
        )

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]
        text_hidden_token = clip_output.text_model_output.hidden_states[-1]

        image_output = self.build_post_processing_module(
            name="image", x=image_hidden_token
        )
        text_output = self.build_post_processing_module(
            name="text", x=text_hidden_token
        )
        self.is_built = True
        log.info(
            f"Built {self.__class__.__name__}, \
                image output: {image_output.shape}, text output: {text_output.shape}"
        )

    def preprocess_image(self, image: torch.Tensor):
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image.unbind(0)
        image = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        image = image.to(self.model.device)

        if len(image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                "method forward_image must be 4, instead it is "
                f"{len(image.shape)}, for shape {image.shape}"
            )
        return image

    def preprocess_text(self, text: torch.Tensor) -> torch.Tensor:
        text = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        text = text.to(self.model.device)
        text = text.to(torch.int32)
        return text

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self.preprocess_image(image)
        clip_output = self.model.forward(pixel_values=image)

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        image_output = self.apply_post_processing(
            name="image", x=image_hidden_token
        )
        return image_output

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:
        text = self.preprocess_text(text)
        if len(text.shape) == 1:
            text = text.unsqueeze(0)
        clip_output = self.model.forward(input_ids=text)

        text_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        text_output = self.apply_post_processing(
            name="text", x=text_hidden_token
        )
        return text_output

    def forward(
        self,
        batch: ImageTextRetrievalInput,
        step: bool = False,
        accelerator: Optional[accelerate.Accelerator] = None,
        **kwargs,
    ) -> CLIPOutput:
        image = batch.target_image[0]
        challenge_images = batch.challenge_images[0]
        images = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch.target_text[0]

        challenge_images = self.preprocess_image(images)
        prompt_text = self.preprocess_text(text)

        if len(prompt_text.shape) == 1:
            prompt_text = prompt_text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=prompt_text,
            pixel_values=challenge_images,
            output_hidden_states=True,
            return_loss=False,
        )

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]
        text_hidden_token = clip_output.text_model_output.hidden_states[-1]

        image_output = self.apply_post_processing(
            name="image", x=image_hidden_token
        )
        text_output = self.apply_post_processing(
            name="text", x=text_hidden_token
        )

        if accelerator is not None:
            image_output = accelerator.gather(image_output)
            text_output = accelerator.gather(text_output)

        similarity = (
            torch.matmul(text_output, image_output.t()) * self.model.logit_scale
        )

        loss = contrastive_loss(similarity)

        if step:
            return self.compute_loss_and_accuracy(
                CLIPModelOutput(
                    logits_per_image=similarity,
                    image_embeds=image_output,
                    text_embeds=text_output,
                    loss=loss,
                )
            )
        return CLIPModelOutput(
            logits_per_image=similarity,
            image_embeds=image_output,
            text_embeds=text_output,
            loss=loss,
        )


# create a main method that runs a test on CLIPImageTextModel and CLIPWithPostProcessingImageTextModel

if __name__ == "__main__":
    dummy_inputs = ImageTextRetrievalInput(
        target_image=torch.rand(1, 3, 224, 224),
        challenge_images=torch.rand(1, 10, 3, 224, 224),
        challenge_paths=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        target_text=["a picture of a cat"],
        collection_images=torch.rand(1, 10, 3, 224, 224),
        collection_paths=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )
    model_name_or_path = "openai/clip-vit-large-patch14"
    model = CLIPImageTextModel(
        model_name_or_path=model_name_or_path, pretrained=True
    )
    accelerator = accelerate.Accelerator()
    model.build(batch=dummy_inputs, accelerator=accelerator)

    model, dummy_inputs = accelerator.prepare(model, dummy_inputs)
    out = model.forward(batch=dummy_inputs, accelerator=accelerator, step=True)
    # print(out)

    dummy_inputs = ImageTextRetrievalInput(
        target_image=torch.rand(1, 3, 224, 224),
        challenge_images=torch.rand(1, 10, 3, 224, 224),
        challenge_paths=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        target_text=["a picture of a cat"],
        collection_images=torch.rand(1, 10, 3, 224, 224),
        collection_paths=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )
    model_name_or_path = "openai/clip-vit-large-patch14"
    model = CLIPWithPostProcessingImageTextModel(
        model_name_or_path=model_name_or_path, pretrained=True
    )
    accelerator = accelerate.Accelerator()
    model.build(batch=dummy_inputs, accelerator=accelerator)

    model, dummy_inputs = accelerator.prepare(model, dummy_inputs)
    out = model.forward(batch=dummy_inputs, accelerator=accelerator, step=True)
    # print(out)
