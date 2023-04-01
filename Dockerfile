FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt upgrade -y

RUN apt install aptitude tree -y
RUN apt install fish -y
RUN apt install wget -y
RUN apt install git -y
RUN apt-get install git -y



RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda

SHELL ["/opt/conda/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0

RUN conda create -n main python=3.10 -y

SHELL ["/opt/conda/bin/conda", "run", "-n", "main", "/bin/bash", "-c"]

RUN conda install -c conda-forge mamba -y
RUN mamba install -c conda-forge jupyterlab black git-lfs -y

RUN mamba install -c conda-forge ffmpeg starship -y
RUN mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
RUN mamba install -c huggingface transformers
RUN echo y | pip install itables torchtyping orjson gradio ipywidgets neptune wandb --upgrade
RUN echo y | pip install accelerate datasets hydra_zen timm pyarrow ftfy --upgrade
RUN echo y | pip install git+https://github.com/BayesWatch/bwatchcompute.git

RUN conda init bash
RUN conda init fish
RUN mamba init bash
RUN mamba init fish

RUN git lfs install

ADD capit/ /app/capit/
ADD setup.py /app/
RUN echo y | pip install /app/

RUN git config --global credential.helper store
RUN git config --global --add safe.directory /app/

ENTRYPOINT ["/bin/bash"]