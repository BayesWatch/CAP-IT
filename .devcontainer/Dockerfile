FROM ghcr.io/bayeswatch/minimal-ml-template:latest

RUN mamba update -c conda-forge ffmpeg starship -y
RUN echo y | pip install itables torchtyping tensorflow tensorflow-datasets

RUN rm -rf /app
ADD . /app
RUN ls /app/
RUN echo y | pip install /app/

RUN git config --global credential.helper store
RUN git config --global --add safe.directory /app/

ENTRYPOINT ["/bin/bash"]
