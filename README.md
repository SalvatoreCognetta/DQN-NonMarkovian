# Reasoning Agent Project
## Clone the project
`git clone https://github.com/SalvatoreCognetta/reasoning-agent-project.git && cd reasoning-agent-project`

## Setup the project

### Pull with all the submodules
`git submodule update --init --recursive`

The submodule is always set to have its HEAD detached at a given commit by default : as the main repository is not tracking the changes of the submodule, it is only seen as a specific commit from the submodule repository.

In order to update an existing Git submodule, you need to execute the “git submodule update” with the “–remote” and the “–merge” option.

`git submodule update --remote --merge`

### Install conda
`sudo bash conda_setup.sh`

### Create conda env with python and pytorch (cpu) installed
`conda create -y --name raenv python=3.8.5`

### Activate env
`conda activate raenv`

### Install tensorforce
`pip install tensorforce==0.6.5`

### Install sapientino-case
`git clone https://github.com/cipollone/gym-sapientino-case.git && cd gym-sapientino-case && pip install .`

### Install Lydia
Make sure to have [Lydia](https://github.com/whitemech/lydia) 
installed on your machine.
We suggest the following setup:

- [Install Docker](https://www.docker.com/get-started)
- Download the Lydia Docker image:
```
docker pull whitemech/lydia:latest
```
- Make the Docker image executable under the name `lydia`.
  On Linux and MacOS machines, the following commands should work:
```
echo '#!/usr/bin/env sh' > lydia
echo 'docker run -v$(pwd):/home/default whitemech/lydia lydia "$@"' >> lydia
sudo chmod u+x lydia
sudo mv lydia /usr/local/bin/
```

This will install an alias to the inline Docker image execution
in your system PATH. Instead of `/usr/local/bin/`
you may use another path which is still in the `PATH` variable.

## Error Numpy
If you get an error like this:
**TypeError: concatenate() got an unexpected keyword argument 'dtype'
**
The problem is the unsupported numpy version, upgrading would produce tensorflow and tensorforce incompatibility. Comment line 205 of the file:
`nano $HOME/.conda/envs/raenv/lib/python3.8/site-packages/gym_sapientino/wrappers/observations.py`



## Train the net
`python main.py --exploration=0.3 --num_colors=3`

To see examples please look at last year projects (similar, though not identical, to this year ones): [Drive](https://docs.google.com/spreadsheets/d/1r5HyGsLVW7F7E2ypZZZkaYBTEF6PJT6hNyIuDOBHbSo/edit#gid=0)

