# Reasoning Agent Project

## Authors
- Appetito Daniele (<appetito.1916560@studenti.uniroma1.it>)
- Cognetta Salvatore (<cognetta.1874383@studenti.uniroma1.it>)
- Rossetti Simone (<rossetti.1900592@studenti.uniroma1.it>)

## Introduction
Lorem ipsum

## Gym SapientinoCase
Lorem ipsum
# Setup the project

## Clone the repository
Open a terminal and download the repo via git:
```bash
git clone https://github.com/SalvatoreCognetta/reasoning-agent-project.git  
cd reasoning-agent-project
```  

## Install conda
Install miniconda with the bash script:
```bash
sudo bash conda_setup.sh
```

## Create conda virtual environment and install packages
### Create the env 
Create the virtual environment with python and pytorch (cpu) installed:  
```bash
conda create -y --name raenv python=3.8.5
```

Activate the environment:  
```bash
conda activate raenv
```

Install tensorforce:  
```bash
pip install tensorforce==0.6.5
```

Install sapientino-case: 
```bash
cd gym-sapientino-case-master && pip install .
```

### Install Lydia
Make sure to have [Lydia](https://github.com/whitemech/lydia) 
installed on your machine.
We suggest the following setup:

- [Install Docker](https://www.docker.com/get-started)
- Download the Lydia Docker image:
```bash
docker pull whitemech/lydia:latest
```
- Make the Docker image executable under the name `lydia`.
  On Linux and MacOS machines, the following commands should work:
```bash
echo '#!/usr/bin/env sh' > lydia
echo 'docker run -v$(pwd):/home/default whitemech/lydia lydia "$@"' >> lydia
sudo chmod u+x lydia
sudo mv lydia /usr/local/bin/
```

This will install an alias to the inline Docker image execution
in your system PATH. Instead of `/usr/local/bin/`
you may use another path which is still in the `PATH` variable.

## Run the project
In order to train the net:  
`python main.py --exploration=0.3 --num_colors=3`

---
## Known Erros: Numpy
If you get an error like this:
**TypeError: concatenate() got an unexpected keyword argument 'dtype'
**
The problem is the unsupported numpy version, upgrading would produce tensorflow and tensorforce incompatibility. Comment line 205 of the file:
```bash
nano $HOME/.conda/envs/raenv/lib/python3.8/site-packages/gym_sapientino/wrappers/observations.py
```
---

# Modification on temprl framework
## Clone temprl project for synthetic experience
In order to modify the temprl repo (used by gym-sapientino-case), before the installation of this repo, clone temprl in another directory:  
```bash
git clone git@github.com:SalvatoreCognetta/temprl.git 
cd temprl
git checkout develop
```

For the code look at the [GitHub repository](https://github.com/SalvatoreCognetta/temprl/tree/develop).

## Modify TemporalGoalWrapper class
Changes are done in `TemporalGoalWrapper` class inside temprl/wrapper.py.  
After a modification on the temprl forked project:
1. push the modifications;
2. remove the directories of temprl inside conda env via: `rm -rvf /home/NAME_TO_CHAGE/anaconda3/envs/raenv/lib/python3.8/site-packages/temprl*` (bug of poetry: [virtual env not updates]([https://link](https://github.com/python-poetry/poetry/issues/2921) ))
3. reinstall gym-sapientino via `cd gym-sapientino-case-master && pip install .`

---
## References
- Sutton, Richard S. and Barto, Andrew G. 2018.Reinforcement Learning: An Introduction(second ed.). <https://mitpress.mit.edu/books/reinforcement-learning-second-edition>.
- Schulman, J. and Wolski, F. and Dhariwal, P. and Radford, A. and Klimov, O. 2017. Proximal policy optimization algorithms <https://arxiv.org/abs/1707.06347>
- Icarte, R. T. and Klassen, T. Q. and Valenzano, R. and McIlraith, S. A. 2020.   Reward machines: Exploiting reward function structure in reinforcement learning. <https://arxiv.org/abs/2010.03950>