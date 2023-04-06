import pathlib
from typing import Optional, Union
import logging
import pathlib
from asyncio.log import logger
from collections import defaultdict
from math import floor
from typing import Optional, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from capit.core.utils.storage import load_json
from capit.core.data.datasets_old import (
    ChallengeSamplesSourceTypes,
    ImageTextRetrievalInput,
    SplitType,
    get_ranked_filepaths_from_user,
)

from capit.decorators import configurable, get_next_on_error
from capit.utils import get_logger
from transformers import CLIPProcessor


logger = get_logger(__name__, logging_level=logging.INFO)


@configurable
class InstagramImageTextMultiModalDatasePyArrow(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, pathlib.Path],
        set_name: str = SplitType.TRAIN,
        top_k_percent: int = 50,
        reset_cache: bool = False,
        num_episodes: int = 100,
        limit_num_samples: Optional[int] = None,
        max_num_collection_images_per_episode: int = 50,
        max_num_query_images_per_episode: int = 50,
        challenge_image_source: str = ChallengeSamplesSourceTypes.WITHIN_USER,
        restrict_num_users: Optional[int] = None,
        dummy_batch_mode: bool = False,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        preprocess: bool = False,
    ):
        super(InstagramImageTextMultiModalDatasePyArrow, self).__init__()

        self.set_name = set_name
        self.dataset_root_dir = pathlib.Path(dataset_dir)
        self.data_source = self.dataset_root_dir
        self.table_source = self.dataset_root_dir / "instagram_table"
        self.reset_cache = reset_cache
        self.num_episodes = num_episodes
        self.limit_num_samples = limit_num_samples
        self.max_num_collection_images_per_episode = (
            max_num_collection_images_per_episode
        )
        self.max_num_query_images_per_episode = max_num_query_images_per_episode
        self.query_image_source = challenge_image_source.lower()
        self.restrict_num_users = restrict_num_users
        self.top_k_percent = top_k_percent
        self.model_name_or_path = model_name_or_path

        self.username_list = list(
            folderpath.name for folderpath in self.table_source.iterdir()
        )

        self.total_num_users = len(self.username_list)
        self.dummy_batch_mode = dummy_batch_mode

        if dummy_batch_mode:
            self.dummy_cache_size = 1
            self.dummy_cache = []

        set_name_to_ratio = {
            SplitType.TRAIN: floor(0.8 * self.total_num_users),
            SplitType.VAL: floor(0.1 * self.total_num_users),
            SplitType.TEST: self.total_num_users - floor(0.9 * self.total_num_users),
        }

        self.set_preprocess_mode(preprocess)

        if set_name == SplitType.TRAIN:
            start_idx = 0
            end_idx = int(set_name_to_ratio[SplitType.TRAIN])
            self.set_usernames = self.username_list[start_idx:end_idx]

        elif set_name == SplitType.VAL:
            start_idx = int(set_name_to_ratio[SplitType.TRAIN])
            end_idx = int(
                set_name_to_ratio[SplitType.TRAIN] + set_name_to_ratio[SplitType.VAL]
            )
            self.set_usernames = self.username_list[start_idx:end_idx]

        elif set_name == SplitType.TEST:
            start_idx = int(
                set_name_to_ratio[SplitType.TRAIN] + set_name_to_ratio[SplitType.VAL]
            )

            self.set_usernames = self.username_list[start_idx:]

    def set_preprocess_mode(self, preprocess: bool):
        self.preprocess = preprocess
        if preprocess:
            self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
                self.model_name_or_path
            )
        return self.preprocess

    def read_image_caption(self, image_path: pathlib.Path, info_path: pathlib.Path):
        if isinstance(image_path, str):
            image_path = pathlib.Path(image_path)

        if isinstance(info_path, str):
            info_path = pathlib.Path(info_path)

        image_path = str(image_path.as_posix()).replace(
            "/data", self.data_source.as_posix()
        )
        image_path = pathlib.Path(image_path)

        info_path = info_path.as_posix().replace("/data", self.data_source.as_posix())

        info_path = pathlib.Path(info_path)

        text = load_json(info_path)["edge_media_to_caption"]["edges"][0]["node"]["text"]

        image = Image.open(image_path)

        return image, text

    @get_next_on_error
    def __getitem__(self, index):
        try:
            if self.dummy_batch_mode and len(self.dummy_cache) >= self.dummy_cache_size:
                return self.dummy_cache[index % self.dummy_cache_size]

            if self.restrict_num_users is not None:
                actual_index = index % self.restrict_num_users
            else:
                actual_index = index % len(self.set_usernames)

            self.current_index = index

            user_name = self.set_usernames[actual_index]
            rng = np.random.RandomState(seed=index)
            user_posts = get_ranked_filepaths_from_user(
                username_filepath=self.table_source / user_name,
                top_k_percent_to_return=self.top_k_percent,
            )

            if len(user_posts) == 0:
                logger.debug("No challenge posts found for this episode")
                raise ValueError("No challenge posts found for this episode")

            target_post_idx = rng.choice(len(user_posts), size=1)[0]
            target_image_path, target_info_path = user_posts[target_post_idx]

            del user_posts[target_post_idx]  # remove target post from collection

            rng.shuffle(user_posts)

            if len(user_posts) == 0:
                logger.debug("No challenge posts found for this episode")
                raise ValueError("No challenge posts found for this episode")

            num_collection_posts = min(
                self.max_num_collection_images_per_episode,
                len(user_posts),
            )
            collection_posts = user_posts[:num_collection_posts]

            if (
                len(collection_posts) == 0
                and self.max_num_collection_images_per_episode > 0
            ):
                logger.debug("No collection posts found for this episode")
                raise ValueError("No collection posts found for this episode")

            if self.query_image_source == ChallengeSamplesSourceTypes.WITHIN_USER:
                num_remaining_user_posts = len(user_posts) - num_collection_posts
                if num_remaining_user_posts <= 1:
                    raise ValueError(
                        "Not enough challenge posts found for this episode"
                    )

                if num_remaining_user_posts == 0:
                    raise ValueError("No challenge posts found for this episode")

                num_challenge_posts = min(
                    self.max_num_query_images_per_episode,
                    num_remaining_user_posts,
                )
                challenge_posts = user_posts[
                    num_collection_posts : num_collection_posts + num_challenge_posts
                ]

                while len(challenge_posts) < self.max_num_query_images_per_episode:
                    challenge_posts = (
                        challenge_posts
                        + user_posts[
                            : self.max_num_query_images_per_episode
                            - len(challenge_posts)
                        ]
                    )

            elif self.query_image_source == ChallengeSamplesSourceTypes.ACROSS_USERS:
                random_user_name_idx = rng.randint(1, len(self.set_usernames))
                while self.set_usernames[random_user_name_idx] == user_name:
                    random_user_name_idx = rng.randint(1, len(self.set_usernames))

                challenge_user_name = self.set_usernames[random_user_name_idx]

                challenge_user_posts = get_ranked_filepaths_from_user(
                    username_filepath=self.table_source / challenge_user_name,
                    top_k_percent_to_return=self.top_k_percent,
                )

                if len(challenge_user_posts) == 0:
                    logger.debug("No challenge posts found for this episode")
                    raise ValueError("No challenge posts found for this episode")

                num_challenge_posts = min(
                    self.max_num_query_images_per_episode,
                    len(challenge_user_posts),
                )
                challenge_posts = challenge_user_posts[:num_challenge_posts]

                if num_remaining_user_posts < self.max_num_query_images_per_episode:
                    challenge_posts = (
                        challenge_posts
                        + challenge_user_posts[
                            : self.max_num_query_images_per_episode
                            - num_remaining_user_posts
                        ]
                    )

            if len(challenge_posts) == 0:
                logger.debug("No challenge posts found for this episode")
                raise ValueError("No challenge posts found for this episode")

            data_dict = defaultdict(list)

            image, text = self.read_image_caption(target_image_path, target_info_path)
            data_dict["target_image"] = image
            data_dict["target_text"] = text
            data_dict["target_text_str"] = text[:300]

            for image_path, info_path in collection_posts:
                image, text = self.read_image_caption(image_path, info_path)
                data_dict["collection_images"].append(image)
                data_dict["collection_paths"].append(
                    dict(image=image_path, info=info_path)
                )

            if (
                len(data_dict["collection_images"]) == 0
                and self.max_num_collection_images_per_episode > 0
            ):
                logger.debug("No collection posts found for this episode")
                raise ValueError("No collection posts found for this episode")
            try:
                for image_path, info_path in challenge_posts:
                    image, text = self.read_image_caption(image_path, info_path)
                    data_dict["challenge_images"].append(image)
                    data_dict["challenge_paths"].append(
                        dict(image=image_path, info=info_path)
                    )
            except Exception as e:
                raise ValueError("No challenge posts found for this episode")

            if len(data_dict["challenge_images"]) == 0:
                logger.debug("No challenge posts found for this episode")
                raise ValueError("No challenge posts found for this episode")

            if (
                len(data_dict["collection_images"]) == 0
                and self.max_num_collection_images_per_episode > 0
            ):
                logger.exception("No collection posts found for this episode")
                raise ValueError("No collection posts found for this episode")

            if self.dummy_batch_mode and len(self.dummy_cache) < self.dummy_cache_size:
                self.dummy_cache.append(ImageTextRetrievalInput(**data_dict))

            output = ImageTextRetrievalInput(**data_dict)

            if (
                output.collection_images is not None
                and len(output.collection_images) > 0
            ):
                if self.preprocess:
                    output.collection_images = self.processor(
                        images=output.collection_images, return_tensors="pt"
                    )["pixel_values"]

            if self.preprocess:
                output.challenge_images = self.processor(
                    images=output.challenge_images, return_tensors="pt"
                )["pixel_values"]

                output.target_text = self.processor(
                    text=output.target_text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]
                output.target_image = self.processor(
                    images=output.target_image, return_tensors="pt"
                )["pixel_values"][0]

            return output
        except Exception as e:
            logger.exception(f"Error in episode {index}, {e}")
            return self.__getitem__(index + 1)

    def __len__(self):
        return self.num_episodes

    def get_user_name_to_post_count_dict(self):
        return {
            user_name: len(pq.read_table(self.table_source / user_name).to_pandas())
            for user_name in self.set_usernames
        }


# write a test for the above class
if __name__ == "__main__":

    def test():
        train_dataset = InstagramImageTextMultiModalDatasePyArrow(
            dataset_dir="/data/",
            set_name="train",
            max_num_collection_images_per_episode=100,
            max_num_query_images_per_episode=100,
            challenge_image_source=ChallengeSamplesSourceTypes.WITHIN_USER,
            top_k_percent=100,
            num_episodes=1000,
        )

        for idx, sample in enumerate(train_dataset):
            for key, value in sample.__dict__.items():
                print(
                    f"{key}: {(value.mean(), value.std(), value.max(), value.min()) if hasattr(value, 'shape') and value.dtype != torch.long else None}"
                )

            if idx > 5:
                break

    test()
