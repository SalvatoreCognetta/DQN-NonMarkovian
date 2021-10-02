import argparse

from tensorforce.environments import Environment

import matplotlib.pyplot as plt

import numpy as np

#Sapientino package
from gym.wrappers import TimeLimit
import os

#Custom observation wrapper for the gymsapientino environment
from gym_sapientino_case.env import SapientinoCase
from gym_sapientino.core.actions import ContinuousCommand
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)

from utils import colors2reward_ldlf
from agent_config import  build_agent
from NonMarkovianTrainer import NonMarkovianTrainer

from argparse import ArgumentParser
import build_dqn

# Constants
MIN_NUM_COLORS = 2
MAX_NUM_COLORS = 5
NUM_COLORS_LIST = [i for i in range(MIN_NUM_COLORS, MAX_NUM_COLORS)]
COLORS_LIST = ['blue','red','yellow','green']
assert len(COLORS_LIST)+1 == MAX_NUM_COLORS

SINK_ID = 2
DEBUG = False


if __name__ == '__main__':
    # Handle command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 500,help= 'Experience batch size.')
    parser.add_argument('--memory', type = int, default = None,help= 'Memory buffer size. Used by agents that train with replay buffer.')
    parser.add_argument('--multi_step',type = int, default = 1, help="Agent update optimization steps.")
    parser.add_argument('--update_frequency', type = int, default = None, help="Frequency of the policy updates. Default equals to batch_size.")
    parser.add_argument('--num_colors', type = int, default = 2, help="Number of distinct colors in the map.")
    parser.add_argument('--learning_rate', type = float, default = 0.001, help="Learning rate for the optimization algorithm")
    parser.add_argument('--exploration', type = float, default = 0.0, help = "Exploration for the epsilon greedy algorithm.")
    parser.add_argument('--entropy_bonus', type = float, default = 0.0, help ="Entropy bonus for the 'extended' loss of PPO. It discourages the policy distribution from being “too certain” (default: no entropy regularization." )
    parser.add_argument('--hidden_size', type = int, default = 64, help="Number of neurons of the hidden layers of the network.")
    parser.add_argument('--max_timesteps', type = int, default = 500, help= "Maximum number of timesteps each episode.")
    parser.add_argument('--episodes', type = int, default = 1000, help = "Number of training episodes.")
    parser.add_argument('--path', type = str, default = None, help = "Path to the map file inside the file system.")
    parser.add_argument("--sequence", nargs="+", default=None, help="Goal sequence for the training specified as a list of strings.")

    args = parser.parse_args()

    # Collect some information from the argument parser.
    batch_size 	= args.batch_size
    memory 		= args.memory
    multi_step 	= args.multi_step
    num_colors 	= args.num_colors
    update_frequency = args.update_frequency
    learning_rate 	 = args.learning_rate
    entropy_bonus 	 = args.entropy_bonus
    exploration 	 = args.exploration
    
    NUM_EXPERTS = num_colors
    EPISODES    = args.episodes
    HIDDEN_STATE_SIZE = args.hidden_size

    # Set this value here to the maximum timestep value.
    MAX_EPISODE_TIMESTEPS = args.max_timesteps
    # There are both the initial and the sink additional states.
    NUM_STATES_AUTOMATON = num_colors+2


    # Extract the map from the command line arguments
    if not args.path:
        if num_colors in NUM_COLORS_LIST:
            map_path = 'maps/map' + str(num_colors) + '_easy.txt'
            map_file = os.path.join('.', map_path)
        else:
            raise AttributeError('Map with ', num_colors,' colors not supported by default. Specify a path for a map file.')
    else:
        map_file = args.path
    
    # Read the txt map file
    with open(map_file) as f:
        map = """""".join(f.readlines())
    # Show in command line
    print(map)

    #Extract the goal sequence form the command line arguments
    if not args.sequence:
        if num_colors == 2:
            colors = ['blue','green']
        elif num_colors == 3:
            colors = ['blue','red','green']
        elif num_colors == 4:
            colors = ['blue','red','yellow','green']
        else:
            raise AttributeError('Map with ', num_colors,' colors not supported by default. Specify a path for a map file.')
    else:
        colors = args.sequence


    # Convert colors in Linear Dynamic Logic 
    reward_ldlf = colors2reward_ldlf(colors) 
    # Show in command line
    print(reward_ldlf)

    # Log directory for the automaton states.
    log_dir = os.path.join('.','log_dir')

    # Istantiate the gym sapientino environment.
    agent_conf = SapientinoAgentConfiguration(
        initial_position=(2, 2),
        commands=ContinuousCommand,
        angular_speed=30.0,
        acceleration=0.10,
        max_velocity=0.40,
        min_velocity=0.0,
    )

    conf = SapientinoConfiguration(
        agent_configs=(agent_conf,),
        grid_map=map,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
        reward_per_step=-0.1,
    )

    environment = SapientinoCase(
        conf=conf,
        reward_ldlf=reward_ldlf,
        logdir=log_dir,
    )

    
    # Default tensorforce update frequency is batch size.
    if not update_frequency:
        update_frequency = batch_size

    # Default dqn memory.
    if not memory:
        memory = 32500 #Replay memory capacity, has to fit at least maximum batch_size + maximum network/estimator horizon + 1 timesteps  #'minimum'


    # Choose whether or not to visualize the environment
    VISUALIZE = True

    # Limit the length of the episode of gym sapientino.
    environment = TimeLimit(environment, MAX_EPISODE_TIMESTEPS)
    # environment.env_synthetic = TimeLimit(environment.env_synthetic, MAX_EPISODE_TIMESTEPS)
    # environment = Environment.create(environment =environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize =VISUALIZE)

    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON

    discount_factor = 0.99

    agent = build_agent(agent = 'dqn', batch_size = batch_size,
                        memory =memory,
                        update_frequency=update_frequency,
                        discount_factor = discount_factor,
                        learning_rate=learning_rate,
                        environment = environment,
                        num_states_automaton =NUM_STATES_AUTOMATON,
                        automaton_state_encoding_size=AUTOMATON_STATE_ENCODING_SIZE,
                        hidden_layer_size=HIDDEN_STATE_SIZE,
                        exploration =exploration,
                        entropy_regularization=entropy_bonus,
                        )

    

    # Debugging prints
    print("Istantiated an agent for training with parameters: ")
    print(args)

    print("The goal sequence is: ")
    print(colors)

    # Create the trainer
    trainer = NonMarkovianTrainer(agent,environment,NUM_STATES_AUTOMATON, 
                                  AUTOMATON_STATE_ENCODING_SIZE,
                                  SINK_ID,num_colors=num_colors
                                  )

    # Train the agent
    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)