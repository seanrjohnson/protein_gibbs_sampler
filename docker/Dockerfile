FROM pytorch/pytorch:latest

ARG workspace=/workspace
ENV TORCH_HOME=${workspace}/cache/torch
ENV PYTORCH_TRANSFORMERS_CACHE=${workspace}/cache/torch_transformers
WORKDIR ${workspace}

RUN apt update && apt upgrade -y && \
    apt install curl vim git -y

ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

RUN conda install -c bioconda hmmer
RUN conda install -c bioconda mafft

# For jupyter lab
EXPOSE 8888

# For tensorboard
EXPOSE 6006