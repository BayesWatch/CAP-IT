import copy
import itertools
import os
from pathlib import Path
from typing import List
from rich import print
import datetime


def get_scripts(exp_name: str, batch_sizes: List[int]):

    script_list = []
    for batch_size in batch_sizes:
        current_script_text = f"/opt/conda/envs/main/bin/accelerate-launch --mixed_precision=bf16 /app/capit/run.py exp_name={exp_name} dataset.max_num_query_images_per_episode=50"
        script_list.append(current_script_text)

    return script_list


if __name__ == "__main__":
    from bwatchcompute.kubernetes import Job, ExperimentTemplate

    script_list = get_scripts(
        exp_name=os.getenv("EXPERIMENT_NAME_PREFIX"), batch_sizes=[75]
    )
    # write a one liner that picks up date and time and converts them into a number
    datetime_seed = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    exp = Job(
        name=f"{datetime_seed}-{os.getenv('EXPERIMENT_NAME_PREFIX')}",
        script_list=script_list,
        docker_image_path=os.getenv("DOCKER_IMAGE_PATH"),
        secret_variables={os.getenv("EXPERIMENT_NAME_PREFIX"): "WANDB_API_KEY"},
        environment_variables={
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "HF_USERNAME": os.getenv("HF_USERNAME"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
            "EXPERIMENTS_DIR": os.getenv("EXPERIMENTS_DIR"),
            "EXPERIMENT_DIR": os.getenv("EXPERIMENT_DIR"),
            "DATASET_DIR": os.getenv("DATASET_DIR"),
            "MODEL_DIR": os.getenv("MODEL_DIR"),
            "PROJECT_DIR": os.getenv("PROJECT_DIR"),
            "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM"),
        },
        num_repeat_experiment=3,
        experiment_template=ExperimentTemplate.standard,
        persistent_disk_claim_names_to_mount_dict={"pvc-instait": "/data/"},
    )

    exp.generate_spec_files()
    output = exp.run_jobs()
    print(output)
