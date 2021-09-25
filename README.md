# Reasoning Agent Project
## Clone the project
`git clone https://github.com/SalvatoreCognetta/reasoning-agent-project.git && cd reasoning-agent-project`

## Pull with all the submodules
`git submodule update --init --recursive`

## Install conda
`sudo bash conda_setup.sh`

## Create conda env 
`conda create -y --name raenv python=3.6 pytorch torchvision torchaudio cpuonly -c pytorch`

## Activate env
`conda activate raenv`

### Install sapientino-case
`cd gym-sapientino-case && pip install .`

To see examples please look at last year projects (similar, though not identical, to this year ones): [Drive](https://docs.google.com/spreadsheets/d/1r5HyGsLVW7F7E2ypZZZkaYBTEF6PJT6hNyIuDOBHbSo/edit#gid=0)

