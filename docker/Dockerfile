FROM pytorch/pytorch:latest

ARG workspace=/workspace
ENV TORCH_HOME=${workspace}/cache/torch
ENV PYTORCH_TRANSFORMERS_CACHE=${workspace}/cache/torch_transformers
WORKDIR ${workspace}

RUN apt-get update && apt-get upgrade -y && \
    apt-get install curl vim git -y

#ADD requirements.txt /tmp
ADD ../conda_env.yml /tmp
#RUN pip install -r /tmp/requirements.txt

RUN conda install -y -c conda-forge mamba
RUN mamba env create -y -f /tmp/conda_env.yml


# For jupyter lab
EXPOSE 8888

# For tensorboard
EXPOSE 6006