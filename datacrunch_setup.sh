#!/bin/bash
exec &> /root/logfile.txt

# set up security
apt-get update -y
apt-get install fail2ban -y
apt-get install ufw -y
ufw allow ssh
yes | ufw enable

# install conda
cd /root
HOME=/root

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
bash $HOME/miniconda.sh -b -p $HOME/miniconda
eval "$(/root/anaconda/bin/conda shell.bash hook)" # this line might not be quite right.
conda init

# install protein_gibbs_sampler
git clone --branch esm1b_followup_random_batch https://github.com/seanrjohnson/protein_gibbs_sampler.git

cd protein_gibbs_sampler

conda activate base

conda create -y -f conda_env.yml

# then when you log on:
# conda activate protein_gibbs_sampler
