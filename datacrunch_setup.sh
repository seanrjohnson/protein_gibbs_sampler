#!/bin/bash
exec &> /root/logfile.txt

# set up security
apt-get update -y
apt-get install fail2ban -y
apt-get install ufw -y
ufw allow ssh
yes | ufw enable
apt-get install -y docker.io make

curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
apt-get install -y nvidia-container-runtime
sudo systemctl restart docker

git clone --branch esm1b_followup https://github.com/seanrjohnson/protein_gibbs_sampler.git

cd protein_gibbs_sampler
make run

