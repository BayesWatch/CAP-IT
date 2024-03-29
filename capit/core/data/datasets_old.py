import json
import logging
from multiprocessing import Value
import os
import pathlib
import random
import shutil
import sys
from asyncio.log import logger
from collections import defaultdict
from dataclasses import dataclass
from math import floor
from typing import Any, Dict, List, Optional, Tuple, Union

from torchtyping import TensorType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import tqdm
from hydra_zen import builds, instantiate
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomCrop, Resize, ToTensor
from capit.core.data.config import ImageShape, ModalityConfig
from capit.core.utils.storage import load_json, save_json

from capit.decorators import configurable, get_next_on_error
from capit.utils import get_logger
from torch.utils.data.dataloader import default_collate
from transformers import CLIPProcessor

logger = get_logger(__name__, logging_level=logging.DEBUG)


@dataclass
class SplitType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class MultiModalInput:
    image: Any = None
    audio: Any = None
    video: Any = None
    text: Any = None


def find_filepaths_with_extension(
    dir_path: str, extension: str, limit_num_files: Optional[int]
):
    filepaths = []

    with tqdm.tqdm(total=12000000) as pbar:
        for path in pathlib.Path(dir_path).iterdir():
            if path.suffix == extension and path.is_file():
                filepaths.append(str(path))
                if limit_num_files is not None:
                    if len(filepaths) >= limit_num_files:
                        break
            pbar.update(1)

    return filepaths


def extract_captions_from_file(filepath: str):
    info_dict = load_json(filepath=filepath)
    return info_dict["edge_media_to_caption"]["edges"][0]["node"]["text"]


def check_if_image_has_matching_info_file(image_path: str):
    if isinstance(image_path, pathlib.Path):
        image_path = str(image_path)
    info_file_path = pathlib.Path(
        image_path.replace("image", "info")
    ).with_suffix(".info")
    return info_file_path.exists()


def get_user_and_post_id_from_image_path(image_path: str):
    username, post_id = image_path.split("/")[-1].split("-")
    post_id = post_id.split(".")[0]

    return username, post_id


def generate_post_paths_from_user_name_and_post_id(
    username: str,
    post_id: str,
    post_image_dir: str,
    post_info_dir: str,
):
    image_path = os.path.join(post_image_dir, f"{username}-{post_id}.jpg")
    info_path = os.path.join(post_info_dir, f"{username}-{post_id}.info")

    return image_path, info_path


@configurable
class DummyMultiModalDataset(Dataset):
    def __init__(
        self,
        set_name: str = "dummy",
        num_samples: int = 100,
        using_pre_sampled_split: bool = False,
        dataset_size_identifier: str = "base",
        dataset_name: str = "base",
        modality_config: ModalityConfig = ModalityConfig(),
        rescan_paths: bool = False,
        num_video_frames_per_datapoint: int = 10,
        num_audio_frames_per_datapoint: int = 88200,
        num_audio_sample_rate: int = 44100,
        image_shape: ImageShape = ImageShape(channels=3, width=224, height=224),
        text_context_length: int = 77,
    ):
        super(DummyMultiModalDataset, self).__init__()

        self.set_name = set_name
        self.num_samples = num_samples
        self.using_pre_sampled_split = using_pre_sampled_split
        self.dataset_size_identifier = dataset_size_identifier
        self.dataset_name = dataset_name
        self.modality_config = modality_config
        self.rescan_paths = rescan_paths
        self.num_video_frames_per_datapoint = num_video_frames_per_datapoint
        self.num_audio_frames_per_datapoint = num_audio_frames_per_datapoint
        self.num_audio_sample_rate = num_audio_sample_rate
        self.image_shape = image_shape
        self.text_context_length = text_context_length

        random.seed(0)
        torch.manual_seed(0)
        torch_rng = torch.Generator()

        if modality_config.text:
            self.text = torch.randint(
                0,
                77,
                size=(
                    self.num_samples,
                    77,
                ),
                generator=torch_rng,
            ).int()

        if modality_config.video:
            self.video = (
                torch.randint(
                    low=1,
                    high=255,
                    size=(
                        self.num_samples,
                        num_video_frames_per_datapoint,
                        image_shape.channels,
                        image_shape.height,
                        image_shape.width,
                    ),
                    generator=torch_rng,
                ).float()
                / 255.0
            )

        if modality_config.audio:
            self.audio = (
                torch.randint(
                    low=1,
                    high=255,
                    size=(
                        self.num_samples,
                        2,
                        num_audio_frames_per_datapoint,
                    ),
                    generator=torch_rng,
                ).float()
                / 255.0
            )

        if modality_config.image:
            self.image = (
                torch.randint(
                    low=1,
                    high=255,
                    size=(
                        self.num_samples,
                        image_shape.channels,
                        image_shape.height,
                        image_shape.width,
                    ),
                    generator=torch_rng,
                ).float()
                / 255.0
            )

    @get_next_on_error
    def __getitem__(self, index) -> MultiModalInput:
        actual_index = index % self.num_samples

        data_dict = MultiModalInput()

        if self.modality_config.text:
            data_dict.text = self.text[actual_index]

        if self.modality_config.image:
            data_dict.image = self.image[actual_index]

        if self.modality_config.video:
            data_dict.video = self.video[actual_index]

        if self.modality_config.audio:
            data_dict.audio = self.audio[actual_index]

        data_dict.filepath = f"{self.set_name}-{index}-{actual_index}"

        return data_dict

    def __len__(self) -> int:
        return 10000000


@dataclass
class ChallengeSamplesSourceTypes:
    WITHIN_USER: str = "within_user"
    ACROSS_USERS: str = "across_users"


def rank_user_items_by_clip_score(username_filepath: pathlib.Path):
    user_table = pq.read_table(username_filepath).to_pandas()
    return user_table.sort_values(by="similarity", ascending=False)


def get_ranked_filepaths_from_user(
    username_filepath: pathlib.Path, top_k_percent_to_return: int
):
    ranked_user_items = rank_user_items_by_clip_score(username_filepath)
    ranked_filepath_list = ranked_user_items["filepath"].tolist()
    top_k_percent_to_return = int(
        len(ranked_filepath_list) * top_k_percent_to_return / 100
    )
    return ranked_filepath_list[:top_k_percent_to_return]


@dataclass
class ImageTextRetrievalInput:
    target_image: TensorType["batch_size", "channels", "height", "width"]
    challenge_images: TensorType[
        "batch_size", "num_challenge_images", "channels", "height", "width"
    ]
    target_text: List[str]
    target_text_str: List[str]
    challenge_paths: List[str] = None
    collection_images: Optional[
        TensorType[
            "batch_size", "num_collection_images", "channels", "height", "width"
        ]
    ] = None
    collection_paths: Optional[List[str]] = None


@dataclass
class ImageTextRetrievalInputByUser:
    image: TensorType["batch_size", "channels", "height", "width"]
    text: List[str]
    filepath: List[str]
    user_name: List[str]
    ids: List[int]


@configurable
class InstagramImageTextMultiModalDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        set_name: str = SplitType.TRAIN,
        reset_cache: bool = False,
        num_episodes: int = 100,
        image_transforms: Optional[Any] = None,
        text_transforms: Optional[Any] = None,
        limit_num_samples: Optional[int] = None,
        max_num_collection_images_per_episode: int = 50,
        max_num_query_images_per_episode: int = 50,
        query_image_source: str = ChallengeSamplesSourceTypes.WITHIN_USER,
        restrict_num_users: Optional[int] = None,
    ):
        super(InstagramImageTextMultiModalDataset, self).__init__()

        self.set_name = set_name
        self.dataset_dir = dataset_dir
        self.reset_cache = reset_cache
        self.num_episodes = num_episodes
        self.image_transforms = image_transforms
        self.text_transforms = text_transforms
        self.limit_num_samples = limit_num_samples
        self.max_num_collection_images_per_episode = (
            max_num_collection_images_per_episode
        )
        self.max_num_query_images_per_episode = max_num_query_images_per_episode
        self.query_image_source = query_image_source.lower()
        self.restrict_num_users = restrict_num_users

        post_image_dir = os.path.join(self.dataset_dir, "image")
        post_info_dir = os.path.join(self.dataset_dir, "info")

        if limit_num_samples is not None:
            user_to_post_id_cache_path = os.path.join(
                self.dataset_dir,
                f"user_to_post_id_cache_num_samples={limit_num_samples}.json",
            )
        else:
            user_to_post_id_cache_path = os.path.join(
                self.dataset_dir,
                "user_to_post_id_cache.json",
            )
        if (not os.path.exists(user_to_post_id_cache_path)) or reset_cache:
            logger.info("Caching info and image paths")
            image_paths_source = find_filepaths_with_extension(
                dir_path=post_image_dir,
                extension=".jpg",
                limit_num_files=limit_num_samples,
            )
            user_to_post_dict = defaultdict(list)

            for image_path in tqdm.tqdm(image_paths_source):
                if check_if_image_has_matching_info_file(image_path):
                    user_name, post_id = get_user_and_post_id_from_image_path(
                        image_path
                    )
                    user_to_post_dict[user_name].append(post_id)

            self._user_to_post_dict = user_to_post_dict
            save_json(
                dict_to_store=self._user_to_post_dict,
                filepath=user_to_post_id_cache_path,
                overwrite=True,
            )
        else:
            logger.info("Loading info and image paths from cache")
            self._user_to_post_dict = load_json(
                filepath=user_to_post_id_cache_path
            )

        self._post_image_dir = post_image_dir
        self._post_info_dir = post_info_dir

        self._idx_to_user_name = list(self._user_to_post_dict.keys())
        total_num_users = len(self._idx_to_user_name)

        set_name_to_ratio = {
            SplitType.TRAIN: 0.8 * total_num_users,
            SplitType.VAL: 0.1 * total_num_users,
            SplitType.TEST: 0.1 * total_num_users,
        }
        if set_name == SplitType.TRAIN:
            start_idx = 0
            end_idx = int(set_name_to_ratio[SplitType.TRAIN])
            self._idx_to_user_name = self._idx_to_user_name[start_idx:end_idx]
        elif set_name == SplitType.VAL:
            start_idx = int(set_name_to_ratio[SplitType.TRAIN])
            end_idx = int(
                set_name_to_ratio[SplitType.TRAIN]
                + set_name_to_ratio[SplitType.VAL]
            )
            self._idx_to_user_name = self._idx_to_user_name[start_idx:end_idx]
        elif set_name == SplitType.TEST:
            start_idx = int(
                set_name_to_ratio[SplitType.TRAIN]
                + set_name_to_ratio[SplitType.VAL]
            )

            self._idx_to_user_name = self._idx_to_user_name[start_idx:]

    @get_next_on_error
    def __getitem__(self, index):
        if self.restrict_num_users is not None:
            actual_index = index % self.restrict_num_users
        else:
            actual_index = index % len(self._idx_to_user_name)
        user_name = self._idx_to_user_name[actual_index]
        rng = np.random.RandomState(seed=index)
        post_id = rng.choice(self._user_to_post_dict[user_name])

        image_path, info_path = generate_post_paths_from_user_name_and_post_id(
            username=user_name,
            post_id=post_id,
            post_image_dir=self._post_image_dir,
            post_info_dir=self._post_info_dir,
        )

        data_dict = ImageTextRetrievalInput()

        data_dict.text = load_json(info_path)["edge_media_to_caption"]["edges"][
            0
        ]["node"]["text"]

        data_dict.image = Image.open(image_path)

        if self.image_transforms is not None:
            data_dict.image = self.image_transforms(data_dict.image)

        data_dict.text_source = data_dict.text
        if self.text_transforms is not None:
            data_dict.text = self.text_transforms(data_dict.text)

        data_dict.filepath = dict(image_path=image_path, info_path=info_path)

        data_dict.collection_images = self._get_user_collection_context_images(
            rng=rng, user_name=user_name, target_post_id=post_id
        )

        if len(data_dict.collection_images) == 0:
            return self.__getitem__(index + 1)

        data_dict.query_image_set = self._get_query_images(
            rng=rng, user_name=user_name, target_post_id=post_id
        )

        if len(data_dict.query_image_set) == 0:
            return self.__getitem__(index + 1)

        data_dict.collection_images = torch.stack(data_dict.collection_images)
        data_dict.query_image_set = torch.stack(data_dict.query_image_set)

        return data_dict

    def _get_query_images(self, rng, user_name, target_post_id):
        query_image_set = []
        if self.query_image_source == ChallengeSamplesSourceTypes.WITHIN_USER:
            shuffled_post_ids = rng.choice(
                self._user_to_post_dict[user_name],
                size=min(
                    len(self._user_to_post_dict[user_name]),
                    self.max_num_query_images_per_episode,
                ),
                replace=False,
            )

            if len(shuffled_post_ids) < self.max_num_query_images_per_episode:
                shuffled_post_ids = np.concatenate(
                    shuffled_post_ids,
                    shuffled_post_ids[
                        : self.max_num_query_images_per_episode
                        - len(shuffled_post_ids)
                    ],
                )

            for idx, collection_post_id in enumerate(shuffled_post_ids):
                if collection_post_id != target_post_id:
                    (
                        image_path,
                        info_path,
                    ) = generate_post_paths_from_user_name_and_post_id(
                        username=user_name,
                        post_id=collection_post_id,
                        post_image_dir=self._post_image_dir,
                        post_info_dir=self._post_info_dir,
                    )

                    collection_image = Image.open(image_path)

                    if self.image_transforms is not None:
                        collection_image = self.image_transforms(
                            collection_image
                        )
                    query_image_set.append(collection_image)
        elif (
            self.query_image_source == ChallengeSamplesSourceTypes.ACROSS_USERS
        ):
            shuffled_user_names = rng.choice(
                self._idx_to_user_name,
                size=self.max_num_query_images_per_episode,
                replace=False,
            )

            for idx, collection_user_name in enumerate(shuffled_user_names):
                if collection_user_name != user_name:
                    collection_post_i = rng.choice(
                        len(self._user_to_post_dict[collection_user_name])
                    )
                    collection_post_id = self._user_to_post_dict[
                        collection_user_name
                    ][collection_post_i]

                    (
                        image_path,
                        info_path,
                    ) = generate_post_paths_from_user_name_and_post_id(
                        username=collection_user_name,
                        post_id=collection_post_id,
                        post_image_dir=self._post_image_dir,
                        post_info_dir=self._post_info_dir,
                    )

                    collection_image = Image.open(image_path)

                    if self.image_transforms is not None:
                        collection_image = self.image_transforms(
                            collection_image
                        )
                    query_image_set.append(collection_image)

        else:
            raise ValueError(
                f"Collection source type {self.query_image_source} not supported"
            )

        return query_image_set

    def _get_user_collection_context_images(
        self, rng, user_name, target_post_id
    ):
        collection_images = []

        shuffled_post_ids = rng.choice(
            self._user_to_post_dict[user_name],
            size=min(
                len(self._user_to_post_dict[user_name]),
                self.max_num_collection_images_per_episode,
            ),
            replace=False,
        )

        if len(shuffled_post_ids) < self.max_num_collection_images_per_episode:
            shuffled_post_ids = np.concatenate(
                shuffled_post_ids,
                shuffled_post_ids[
                    : self.max_num_collection_images_per_episode
                    - len(shuffled_post_ids)
                ],
            )

        for idx, collection_post_id in enumerate(shuffled_post_ids):
            if collection_post_id != target_post_id:
                (
                    image_path,
                    info_path,
                ) = generate_post_paths_from_user_name_and_post_id(
                    username=user_name,
                    post_id=collection_post_id,
                    post_image_dir=self._post_image_dir,
                    post_info_dir=self._post_info_dir,
                )

                collection_image = Image.open(image_path)

                if self.image_transforms is not None:
                    collection_image = self.image_transforms(collection_image)
                collection_images.append(collection_image)

        return collection_images

    def get_user_collection(self, index):
        actual_index = index % len(self._idx_to_user_name)
        user_name = self._idx_to_user_name[actual_index]

        data_dict = defaultdict(list)

        for post_id in self._user_to_post_dict[user_name]:
            (
                image_path,
                info_path,
            ) = generate_post_paths_from_user_name_and_post_id(
                username=user_name,
                post_id=post_id,
                post_image_dir=self._post_image_dir,
                post_info_dir=self._post_info_dir,
            )

            text = load_json(info_path)["edge_media_to_caption"]["edges"][0][
                "node"
            ]["text"]

            image = Image.open(image_path)

            if self.image_transforms is not None:
                image = self.image_transforms(image)

            data_dict["image"].append(image)
            data_dict["text"].append(text)
            data_dict["user_name"] = user_name

        return data_dict

    def __len__(self):
        return self.num_episodes

    def get_user_name_to_post_count_dict(self):
        return {
            user_name: len(post_ids)
            for user_name, post_ids in self._user_to_post_dict.items()
        }


class ToThreeChannels(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            return image[:3]
        else:
            return


def default_image_transforms():
    return Compose(
        [Resize(224), RandomCrop(224), ToTensor(), ToThreeChannels()]
    )


default_image_transforms_config = builds(default_image_transforms)


@configurable
class InstagramImageTextMultiModalDatasetByUser(
    InstagramImageTextMultiModalDataset
):
    def __init__(
        self,
        dataset_dir: str,
        set_name: str = SplitType.TRAIN,
        reset_cache: bool = False,
        num_episodes: int = 100,
        image_transforms: Optional[Any] = None,
        text_transforms: Optional[Any] = None,
        limit_num_samples: Optional[int] = None,
        max_num_collection_images_per_episode: int = 50,
        max_num_query_images_per_episode: int = 50,
        query_image_source: str = ChallengeSamplesSourceTypes.WITHIN_USER,
        restrict_num_users: Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            set_name=set_name,
            reset_cache=reset_cache,
            num_episodes=num_episodes,
            image_transforms=image_transforms,
            text_transforms=text_transforms,
            limit_num_samples=limit_num_samples,
            max_num_collection_images_per_episode=max_num_collection_images_per_episode,
            max_num_query_images_per_episode=max_num_query_images_per_episode,
            query_image_source=query_image_source,
            restrict_num_users=restrict_num_users,
        )

    def __len__(self):
        return len(self._idx_to_user_name)

    def _get_user_collection_context_images(self, user_name):
        images = []
        captions = []
        filepaths = []

        for idx, collection_post_id in enumerate(
            self._user_to_post_dict[user_name]
        ):
            (
                image_path,
                info_path,
            ) = generate_post_paths_from_user_name_and_post_id(
                username=user_name,
                post_id=collection_post_id,
                post_image_dir=self._post_image_dir,
                post_info_dir=self._post_info_dir,
            )

            preprocess_transforms = Compose(
                [
                    Resize((224)),
                    RandomCrop((224)),
                    ToTensor(),
                    ToThreeChannels(),
                ]
            )

            image = preprocess_transforms(Image.open(image_path))

            if self.image_transforms is not None:
                image = self.image_transforms(image)

            text = self.get_text_from_filepath(filepath=info_path)

            if self.text_transforms is not None:
                text = self.text_transforms(text)

            if text is not False:
                images.append(image)
                captions.append(text)
                filepaths.append((image_path, info_path))

        return images, captions, filepaths

    @get_next_on_error
    def __getitem__(self, index):
        if self.restrict_num_users is not None:
            actual_index = index % self.restrict_num_users
        else:
            actual_index = index % len(self._idx_to_user_name)
        user_name = self._idx_to_user_name[actual_index]

        images, captions, filepaths = self._get_user_collection_context_images(
            user_name=user_name
        )

        return ImageTextRetrievalInputByUser(
            image=images,
            text=captions,
            filepath=filepaths,
            user_name=[user_name] * len(images),
            ids=list(range(len(images))),
        )

    def get_text_from_filepath(self, filepath):
        try:
            text = load_json(filepath)["edge_media_to_caption"]["edges"][0][
                "node"
            ]["text"]

            return text
        except Exception:
            logger.debug(
                "Could not find valid text for this target image. A sample will be resampled.",
            )

            return False


def dataclass_collate(batch):
    """Collate data from a list of dataclass objects.

    Args:
        batch (list): List of dataclass objects.

    Returns:
        dict: Dictionary of values from the dataclass objects.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if isinstance(batch[0], dict) or not hasattr(
        batch[0], "__dataclass_fields__"
    ):
        return default_collate(batch)
    else:
        batched_dict = {
            key: default_collate([getattr(sample, key) for sample in batch])
            if getattr(batch[0], key) != None
            else None
            for key in batch[0].__dict__.keys()
        }
        return batch[0].__class__(**batched_dict)


class ToThreeChannels(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            return image[:3]
        else:
            return image


def get_image_transforms_instait():
    return Compose([Resize((224, 224)), ToTensor(), ToThreeChannels()])


if __name__ == "__main__":
    from rich import print

    os.environ["HYDRA_FULL_ERROR"] = "1"

    root_filepath = pathlib.Path("/data/")
    table_source = root_filepath / "instagram_table"

    username_list = list(
        folderpath.name for folderpath in table_source.iterdir()
    )
    found_dict = {"found": 0, "not_found": 0}
    with tqdm.tqdm(total=len(username_list)) as pbar:
        for username_filepath in username_list:
            print(username_filepath)
            try:
                user_table = pq.read_table(
                    table_source / username_filepath
                ).to_pandas()
                user_table.sort_values(by="similarity", ascending=False)

            except Exception as e:
                print(f"Could not read table for {username_filepath}: {e}")
                found_dict["not_found"] += 1
                shutil.rmtree(
                    pathlib.Path(table_source / username_filepath).as_posix()
                )
                if pathlib.Path(table_source / username_filepath).exists():
                    print(
                        f"Folder exists after being deleted: {table_source / username_filepath}"
                    )
                continue

            for entry in user_table.iterrows():
                entry_idx, entry = entry
                image_path, info_path = entry["filepath"]
                if (
                    not pathlib.Path(image_path).exists()
                    or not pathlib.Path(info_path).exists()
                ):
                    print(f"Filepath {entry['filepath']} does not exist.")
                    found_dict["not_found"] += 1
                    if pathlib.Path(image_path).exists():
                        pathlib.Path(image_path).unlink()
                    if pathlib.Path(info_path).exists():
                        pathlib.Path(info_path).unlink()
                    parquet_file_path = (
                        table_source
                        / username_filepath
                        / f"{entry['id']}.parquet"
                    )
                    print(
                        f"Parquet file path: {parquet_file_path}. {pathlib.Path(parquet_file_path).exists()} {entry['id']} {entry}"
                    )
                    if pathlib.Path(parquet_file_path).exists():
                        # delete folder parquet file path
                        parquet_file_path.unlink()
                        print(f"Deleted parquet file path: {parquet_file_path}")

                else:
                    found_dict["found"] += 1
            pbar.update(1)
            pbar.set_description(
                f"Found: {found_dict['found']}, Not Found: {found_dict['not_found']}"
            )

    print(found_dict)
