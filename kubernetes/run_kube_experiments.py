import copy
import itertools
import os
from pathlib import Path
from typing import List
from rich import print
import datetime


def get_scripts(
    exp_name: str,
    model_name_list: List[int],
    pretrained_list: List[bool],
    backbone_fine_tunable_list: List[bool],
    optimizer_lr_list: List[float],
    optimizer_weight_decay_list: List[float],
):

    script_list = set()
    for model_name in model_name_list:
        for pretrained in pretrained_list:
            for backbone_fine_tunable in backbone_fine_tunable_list:
                for optimizer_lr in optimizer_lr_list:
                    for optimizer_weight_decay in optimizer_weight_decay_list:

                        if model_name == "clip-baseline":
                            name = f"{exp_name}-{model_name}-{pretrained}-{optimizer_lr}-{optimizer_weight_decay}"
                            current_script_text = (
                                "/opt/conda/envs/main/bin/accelerate-launch "
                                "--mixed_precision=bf16 "
                                "--gradient_accumulation_steps=25 "
                                "/app/capit/run.py "
                                f"exp_name={name} "
                                f"model={model_name} "
                                f"model.pretrained={pretrained} "
                                f"optimizer.lr={optimizer_lr} "
                                f"optimizer.weight_decay={optimizer_weight_decay} "
                                "dataset.max_num_query_images_per_episode=50 "
                                "dataset.top_k_percent=100"
                            )

                        elif model_name == "clip-with-post-processing-baseline":
                            name = f"{exp_name}-{model_name}-{pretrained}-{optimizer_lr}-{optimizer_weight_decay}-{backbone_fine_tunable}"
                            current_script_text = (
                                "/opt/conda/envs/main/bin/accelerate-launch "
                                "--mixed_precision=bf16 "
                                "--gradient_accumulation_steps=25 "
                                "/app/capit/run.py "
                                f"exp_name={name} "
                                f"model={model_name} "
                                f"model.pretrained={pretrained} "
                                f"optimizer.lr={optimizer_lr} "
                                f"optimizer.weight_decay={optimizer_weight_decay} "
                                f"model.backbone_fine_tunable={backbone_fine_tunable} "
                                "dataset.max_num_query_images_per_episode=50 "
                                "dataset.top_k_percent=100"
                            )

                        script_list.add(current_script_text)

    return list(script_list)


if __name__ == "__main__":
    from bwatchcompute.kubernetes import Job, ExperimentTemplate

    script_list = get_scripts(
        exp_name=f"{os.getenv('EXPERIMENT_NAME_PREFIX')}-v1.1",
        model_name_list=["clip-baseline", "clip-with-post-processing-baseline"],
        pretrained_list=[True, False],
        backbone_fine_tunable_list=[True, False],
        optimizer_lr_list=[2e-5, 2e-4],
        optimizer_weight_decay_list=[0.0, 1e-5],
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
        num_repeat_experiment=10,
        experiment_template=ExperimentTemplate.standard,
        persistent_disk_claim_names_to_mount_dict={"pvc-instait3": "/data/"},
    )

    exp.generate_spec_files()
    output = exp.run_jobs()
    print(output)
