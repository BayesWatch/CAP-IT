from dataclasses import dataclass
import math
import pathlib
from typing import Any, Iterator, List, Optional, Tuple, Union
from unittest.util import strclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from rich import print
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput, contrastive_loss
from huggingface_hub import (
    Repository,
    create_repo,
    hf_hub_download,
    login,
    snapshot_download,
)


from capit.core.data.datasets import ImageTextRetrievalInput
from capit.core.models.baselines import (
    CLIPImageTextModel,
    contrastive_accuracy,
    contrastive_accuracy_top_k,
)
from capit.core.utils import get_logger
from capit.decorators import configurable

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        max_len = x.shape[1]
        d_model = x.shape[2]
        position = torch.arange(max_len).unsqueeze(1).to(x.device)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        ).to(x.device)
        pe = torch.zeros(1, max_len, d_model).to(x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term).to(x.device)
        pe[0, :, 1::2] = torch.cos(position * div_term).to(x.device)
        x = x + pe[: x.size(0)]
        return x


@configurable
class SummaryTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        batch_first: bool = True,
        norm_first: bool = True,
        activation: nn.Module = F.gelu,
        output_linear_layer_dim: int = 512,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.activation = activation
        self.output_linear_layer_dim = output_linear_layer_dim
        self.use_positional_encoding = use_positional_encoding

        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            encoder_layer=transformer_layer,
            norm=nn.LayerNorm(d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_linear_weight = nn.Linear(
            d_model, output_linear_layer_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_positional_encoding:
            x = x + self.pos_encoder(x)
        b, s, f = x.shape
        x = self.transformer(x).view(-1, x.shape[-1])
        x = self.output_norm(x)
        x = self.output_linear_weight(x)
        x = x.view(b, s, self.output_linear_layer_dim)
        return x


@dataclass
class CLIPModelOutput:
    logits_per_image: torch.Tensor
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    loss: Optional[torch.Tensor] = None


@configurable
class CAPCLIPImageTextModel(CLIPImageTextModel):
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
        self.accelerator = Accelerator()
        self.is_built = False

    def build_cap_module(
        self,
        x_collection_embeddings: torch.Tensor,
        x_query_embeddings: torch.Tensor,
        x_text_embeddings: torch.Tensor,
    ):
        self.summary_vector_transformer = SummaryTransformer(
            d_model=x_collection_embeddings.shape[-1],
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            num_layers=4,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
            output_linear_layer_dim=512,
            use_positional_encoding=False,
        )
        summary_vector = self.summary_vector_transformer(
            x_collection_embeddings
        )[:, -1, :]

        self.modulation_transformer = SummaryTransformer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            num_layers=4,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
            output_linear_layer_dim=512,
            use_positional_encoding=False,
        )
        self.mixing_factor = nn.Parameter(torch.ones(2), requires_grad=True)
        mixed_tokens = torch.cat(
            [
                summary_vector.unsqueeze(1),
                x_query_embeddings.unsqueeze(0),
                x_text_embeddings.unsqueeze(0),
            ],
            dim=1,
        )
        modulated_tokens = self.modulation_transformer(mixed_tokens)
        x_text_embeddings_cap = modulated_tokens[
            :, 1 : x_text_embeddings.shape[0] + 1, :
        ]
        x_query_embeddings_cap = modulated_tokens[
            :, x_text_embeddings.shape[0] + 1 :, :
        ]

        return {
            "text": x_text_embeddings
            + self.mixing_factor[0] * x_text_embeddings_cap,
            "image": x_query_embeddings_cap * self.mixing_factor[1]
            + x_query_embeddings,
        }

    def apply_cap_module(
        self,
        x_collection_embeddings: torch.Tensor,
        x_query_embeddings: torch.Tensor,
        x_text_embeddings: torch.Tensor,
    ):
        summary_vector = self.summary_vector_transformer(
            x_collection_embeddings
        )[:, -1, :]
        x_query_embeddings_cap = self.modulation_transformer(
            x_query_embeddings.unsqueeze(0) * summary_vector
        )
        x_text_embeddings_cap = self.modulation_transformer(
            x_text_embeddings.unsqueeze(0) * summary_vector
        )

        return {
            "text": x_text_embeddings
            + self.mixing_factor[0] * x_text_embeddings_cap,
            "image": x_query_embeddings_cap * self.mixing_factor[1]
            + x_query_embeddings,
        }

    def build(
        self,
        batch: ImageTextRetrievalInput,
        accelerator: Optional[Accelerator] = None,
    ):
        logger.info(f"Building model {self.__class__.__name__}")
        image = batch.target_image[0]
        challenge_images = batch.challenge_images[0][0].unsqueeze(0)
        image = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch.target_text[0]

        # image = self.preprocess_image(image)
        collection_images = batch.collection_images[0]
        # text = self.preprocess_text(text)

        if len(text.shape) == 1:
            text = text.unsqueeze(0)

        clip_output: CLIPModelOutput = self.model.forward(
            input_ids=text,
            pixel_values=image,
            output_hidden_states=True,
            return_loss=False,
        )
        collection_image_embeddings = self.model.vision_model.forward(
            pixel_values=collection_images,
        )[1].unsqueeze(0)

        personalized_embeddings_dict = self.build_cap_module(
            x_collection_embeddings=collection_image_embeddings,
            x_query_embeddings=clip_output.image_embeds,
            x_text_embeddings=clip_output.text_embeds,
        )

        self.is_built = True
        logger.info(
            f"Built {self.__class__.__name__}, \
                image output: {personalized_embeddings_dict['image'].shape}, text output: {personalized_embeddings_dict['text'].shape}"
        )

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        # image = self.preprocess_image(image)
        clip_output = self.model.forward(pixel_values=image)

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        return image_hidden_token

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:
        # text = self.preprocess_text(text)
        if len(text.shape) == 1:
            text = text.unsqueeze(0)
        clip_output = self.model.forward(input_ids=text)

        text_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        return text_hidden_token

    def forward(
        self,
        batch: ImageTextRetrievalInput,
        accelerator: Accelerator = None,
        step: bool = False,
        **kwargs,
    ) -> CLIPOutput:
        image = batch.target_image[0]
        challenge_images = batch.challenge_images[0]
        collection_images = batch.collection_images[0]
        challenge_images = torch.cat(
            [image.unsqueeze(0), challenge_images], dim=0
        )
        prompt_text = batch.target_text[0]

        # challenge_images = self.preprocess_image(challenge_images)
        # collection_images = self.preprocess_image(collection_images)
        # prompt_text = self.preprocess_text(prompt_text)

        challenge_images = challenge_images.to(self.accelerator.device)
        collection_images = collection_images.to(self.accelerator.device)
        prompt_text = prompt_text.to(self.accelerator.device)

        if len(prompt_text.shape) == 1:
            prompt_text = prompt_text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=prompt_text,
            pixel_values=challenge_images,
            output_hidden_states=True,
            return_loss=False,
        )

        collection_image_embeddings = self.model.vision_model.forward(
            pixel_values=collection_images,
        )[1].unsqueeze(0)

        image_output = clip_output.image_embeds
        text_output = clip_output.text_embeds

        personalized_embeddings_dict = self.apply_cap_module(
            x_collection_embeddings=collection_image_embeddings,
            x_query_embeddings=image_output,
            x_text_embeddings=text_output,
        )

        image_output = personalized_embeddings_dict["image"][
            0
        ]  #  self.model.vision_model.post_layernorm
        text_output = personalized_embeddings_dict["text"][
            0
        ]  # self.model.text_model.final_layer_norm
        # normalized features
        # print(f"text_output: {text_output.shape}, image_output: {image_output.shape}")
        image_output = image_output / image_output.norm(
            p=2, dim=-1, keepdim=True
        )
        text_output = text_output / text_output.norm(p=2, dim=-1, keepdim=True)

        # print(f"text_output: {text_output.shape}, image_output: {image_output.shape}")

        similarity = (
            torch.matmul(text_output, image_output.t())
            * self.model.logit_scale.exp()
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


@configurable
class CAPCLIPWithPostProcessingImageTextModel(CLIPImageTextModel):
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

    def build_cap_module(
        self,
        name: str,
        x: torch.Tensor,
        text_embedding_size: int,
        image_embedding_size: int,
    ):
        transformer_encoder = nn.TransformerEncoderLayer(
            d_model=x.shape[2], nhead=8, dim_feedforward=2048
        )
        encoder_norm = nn.LayerNorm(x.shape[2])
        self.post_processing_module[
            f"{name}_cap_transformer"
        ] = nn.TransformerEncoder(
            encoder_layer=transformer_encoder, num_layers=1, norm=encoder_norm
        )
        x = self.post_processing_module[f"{name}_cap_transformer"](x)
        x = x.mean(dim=1)
        self.post_processing_module[f"{name}-cap-text_output"] = nn.Linear(
            x.shape[1], text_embedding_size
        )
        self.post_processing_module[f"{name}-cap-image_output"] = nn.Linear(
            x.shape[1], image_embedding_size
        )
        out_text = self.post_processing_module[f"{name}-cap-text_output"](x)
        out_image = self.post_processing_module[f"{name}-cap-image_output"](x)
        return {"text": out_text, "image": out_image}

    def apply_post_processing(self, x: torch.Tensor, name: str):
        x = self.post_processing_module[f"{name}_transformer"](x)
        x = x.mean(dim=1)
        x = self.post_processing_module[f"{name}_output"](x)
        return x

    def apply_cap_module(self, x: torch.Tensor, name: str):
        x = self.post_processing_module[f"{name}_cap_transformer"](x)
        x = x.mean(dim=1)

        out_text = self.post_processing_module[f"{name}-cap-text_output"](x)
        out_image = self.post_processing_module[f"{name}-cap-image_output"](x)
        return {"text": out_text, "image": out_image}

    def build(
        self,
        batch: ImageTextRetrievalInput,
        accelerator: Optional[Accelerator] = None,
    ):
        logger.info(f"Building model {self.__class__.__name__}")
        image = batch.target_image[0]
        challenge_images = batch.challenge_images[0][0].unsqueeze(0)
        image = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch.target_text[0]

        # image = self.preprocess_image(image)
        collection_images = batch.collection_images[0]
        # text = self.preprocess_text(text)

        if len(text.shape) == 1:
            text = text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=text,
            pixel_values=image,
            output_hidden_states=True,
            return_loss=False,
        )
        collection_image_embeddings = self.model.vision_model.forward(
            pixel_values=collection_images,
        )[1].unsqueeze(0)

        collection_personalization_vector = self.build_cap_module(
            name="cap-network",
            x=collection_image_embeddings,
            text_embedding_size=clip_output.text_model_output.hidden_states[
                -1
            ].shape[2],
            image_embedding_size=clip_output.vision_model_output.hidden_states[
                -1
            ].shape[2],
        )

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]
        text_hidden_token = clip_output.text_model_output.hidden_states[-1]

        image_hidden_token = torch.cat(
            [
                image_hidden_token,
                collection_personalization_vector["image"]
                .unsqueeze(1)
                .repeat([image_hidden_token.shape[0], 1, 1]),
            ],
            dim=1,
        )

        text_hidden_token = torch.cat(
            [
                text_hidden_token,
                collection_personalization_vector["text"]
                .unsqueeze(1)
                .repeat([text_hidden_token.shape[0], 1, 1]),
            ],
            dim=1,
        )

        image_output = self.build_post_processing_module(
            name="image", x=image_hidden_token
        )
        text_output = self.build_post_processing_module(
            name="text", x=text_hidden_token
        )

        self.is_built = True
        logger.info(
            f"Built {self.__class__.__name__}, \
                image output: {image_output.shape}, text output: {text_output.shape}"
        )

    # def preprocess_image(self, image: torch.Tensor):
    #     if isinstance(image, torch.Tensor):
    #         if len(image.shape) == 4:
    #             image = image.unbind(0)
    #     image = self.processor(images=image, return_tensors="pt")["pixel_values"]
    #     image = image.to(self.model.device)

    #     if len(image.shape) != 4:
    #         raise ValueError(
    #             f"Input shape for class {self.__class__.__name__} in "
    #             "method forward_image must be 4, instead it is "
    #             f"{len(image.shape)}, for shape {image.shape}"
    #         )
    #     return image

    # def preprocess_text(self, text: torch.Tensor) -> torch.Tensor:
    #     text = self.processor(
    #         text=text, return_tensors="pt", padding=True, truncation=True
    #     )["input_ids"]
    #     text = text.to(self.model.device)
    #     text = text.to(torch.int32)
    #     return text

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        # image = self.preprocess_image(image)
        clip_output = self.model.forward(pixel_values=image)

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        image_output = self.apply_post_processing(
            name="image", x=image_hidden_token
        )
        return image_output

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:
        # text = self.preprocess_text(text)
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
        accelerator: Accelerator = None,
        step: bool = False,
        **kwargs,
    ) -> CLIPOutput:
        image = batch.target_image[0]
        challenge_images = batch.challenge_images[0]
        collection_images = batch.collection_images[0]
        challenge_images = torch.cat(
            [image.unsqueeze(0), challenge_images], dim=0
        )
        prompt_text = batch.target_text[0]

        # challenge_images = self.preprocess_image(challenge_images)
        # collection_images = self.preprocess_image(collection_images)
        # prompt_text = self.preprocess_text(prompt_text)

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

        collection_image_embeddings = self.model.vision_model.forward(
            pixel_values=collection_images,
        )[1].unsqueeze(0)

        collection_personalization_vector = self.apply_cap_module(
            name="cap-network", x=collection_image_embeddings
        )

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]
        text_hidden_token = clip_output.text_model_output.hidden_states[-1]

        image_hidden_token = torch.cat(
            [
                image_hidden_token,
                collection_personalization_vector["image"]
                .unsqueeze(1)
                .repeat([image_hidden_token.shape[0], 1, 1]),
            ],
            dim=1,
        )

        text_hidden_token = torch.cat(
            [
                text_hidden_token,
                collection_personalization_vector["text"]
                .unsqueeze(1)
                .repeat([text_hidden_token.shape[0], 1, 1]),
            ],
            dim=1,
        )

        image_output = self.apply_post_processing(
            name="image", x=image_hidden_token
        )
        text_output = self.apply_post_processing(
            name="text", x=text_hidden_token
        )

        if accelerator is not None:
            image_output = accelerator.gather(image_output)
            text_output = accelerator.gather(text_output)

        # normalized features
        image_output = image_output / image_output.norm(
            p=2, dim=-1, keepdim=True
        )
        text_output = text_output / text_output.norm(p=2, dim=-1, keepdim=True)

        similarity = (
            torch.matmul(text_output, image_output.t())
            * self.model.logit_scale.exp()
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

    def compute_loss_and_accuracy(self, clip_output: CLIPModelOutput):
        accuracy = contrastive_accuracy(clip_output.logits_per_image)
        accuracy_top_5 = contrastive_accuracy_top_k(
            clip_output.logits_per_image, 5
        )
        output_dict = clip_output.__dict__
        output_dict["metrics"] = {
            "accuracy": accuracy,
            "accuracy_top_5": accuracy_top_5,
            "loss": clip_output.loss,
        }

        return output_dict["loss"], output_dict


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
    model = CAPCLIPImageTextModel(
        model_name_or_path=model_name_or_path, pretrained=True
    )
    accelerator = Accelerator()
    model.build(batch=dummy_inputs, accelerator=accelerator)

    model, dummy_inputs = accelerator.prepare(model, dummy_inputs)
    out = model.forward(batch=dummy_inputs, accelerator=accelerator, step=True)
    # print(out)
