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
eval "$(/root/miniconda/bin/conda shell.bash hook)"
conda init

# install protein_gibbs_sampler and configure conda environment
git clone --branch metrics https://github.com/seanrjohnson/protein_gibbs_sampler.git

cd protein_gibbs_sampler

conda activate base

conda env create -f conda_env.yml

echo "conda activate protein_gibbs_sampler" >> $HOME/.bashrc

# make a data directory, which will be at: /root/protein_gibbs_sampler/data

mkdir data