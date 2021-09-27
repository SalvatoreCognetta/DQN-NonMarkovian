# Reasoning Agent Project
## Clone the project
`git clone https://github.com/SalvatoreCognetta/reasoning-agent-project.git && cd reasoning-agent-project`

## Setup the project

### Pull with all the submodules
`git submodule update --init --recursive`

### Install conda
`sudo bash conda_setup.sh`

### Create conda env with python and pytorch (cpu) installed
`conda create -y --name raenv python=3.8.5`

### Activate env
`conda activate raenv`

### Install tensorforce
`pip install tensorforce`

### Install sapientino-case
`cd gym-sapientino-case && pip install .`

## Train the net
`python main.py --exploration=0.3 --num_colors=2`

To see examples please look at last year projects (similar, though not identical, to this year ones): [Drive](https://docs.google.com/spreadsheets/d/1r5HyGsLVW7F7E2ypZZZkaYBTEF6PJT6hNyIuDOBHbSo/edit#gid=0)

