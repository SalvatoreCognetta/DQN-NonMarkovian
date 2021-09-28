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

from agent_config import  build_agent

from NonMarkovianTrainer import NonMarkovianTrainer

from argparse import ArgumentParser



SINK_ID = 2

DEBUG = False








if __name__ == '__main__':


    #Handle command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 64,help= 'Experience batch size.')
    parser.add_argument('--memory', type = int, default = None,help= 'Memory buffer size. Used by agents that train with replay buffer.')
    parser.add_argument('--multi_step',type = int, default = 10, help="Agent update optimization steps.")
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

    #Collect some information from the argument parser.
    batch_size = args.batch_size
    memory = args.memory
    update_frequency = args.update_frequency
    multi_step = args.multi_step
    num_colors = args.num_colors
    learning_rate = args.learning_rate
    entropy_bonus = args.entropy_bonus
    exploration = args.exploration




    #Extract the map from the command line arguments
    if not args.path:
        if num_colors == 2:
            map_file = os.path.join('.','maps/map2_easy.txt')
        elif num_colors == 3:
            map_file = os.path.join('.', 'maps/map3_easy.txt')
        elif num_colors == 4:
            map_file = os.path.join('.','maps/map4_easy.txt')
        else:
            raise AttributeError('Map with ', num_colors,' colors not supported by default. Specify a path for a map file.')
    else:
        map_file = args.path


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


    #Log directory for the automaton states.
    log_dir = os.path.join('.','log_dir')

    #Istantiate the gym sapientino environment.
    agent_conf = SapientinoAgentConfiguration(
        initial_position=(2, 2),
        commands=ContinuousCommand,
        angular_speed=30.0,
        acceleration=0.10,
        max_velocity=0.40,
        min_velocity=0.0,
    )

    _default_map = """\
    |           |
    |     b     |
    |     #     |
    |  r  # g   |
    |     #     |
    |###########|"""

    conf = SapientinoConfiguration(
        agent_configs=(agent_conf,),
        grid_map=_default_map,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
        reward_per_step=0.0,
    )

    def colors2reward_ldlf(colors:list): # convert color list to ldlf
        reward_ldlf = "<"
        for i,c in enumerate(colors):
            reward_ldlf+="!"+c+"*; "+c
            if i<len(colors)-1:
                reward_ldlf+="; "
            elif i==len(colors)-1:
                reward_ldlf+=">end"
        return reward_ldlf
    
    reward_ldlf = colors2reward_ldlf(colors) 

    print(reward_ldlf)

    environment = SapientinoCase(
        conf=conf,
        reward_ldlf=reward_ldlf,
        logdir=log_dir,
    )
    # environment = SapientinoCase(

    #     colors = colors,

    #     params = dict(
    #         reward_per_step=-1.0,
    #         reward_outside_grid=0.0,
    #         reward_duplicate_beep=0.0,
    #         acceleration=0.4,
    #         angular_acceleration=15.0,
    #         max_velocity=0.6,
    #         min_velocity=0.4,
    #         max_angular_vel=40,
    #         initial_position=[4, 2],
    #         tg_reward=1000.0,
    #     ),

    #     map_file = map_file,
    #     logdir =log_dir

    # )



    #Default tensorforce update frequency is batch size.
    if not update_frequency:
        update_frequency = batch_size

    #Default dqn memory.
    if not memory:
        memory = 20000 #Replay memory capacity, has to fit at least maximum batch_size + maximum network/estimator horizon + 1 timesteps  #'minimum'

    HIDDEN_STATE_SIZE = args.hidden_size



    #There are both the initial and the sink additional states.
    NUM_STATES_AUTOMATON = num_colors+2
    

    NUM_EXPERTS = num_colors


    #Set this value here to the maximum timestep value.
    MAX_EPISODE_TIMESTEPS = args.max_timesteps

    #Choose whether or not to visualize the environment
    VISUALIZE = True

    # Limit the length of the episode of gym sapientino.
    environment = TimeLimit(environment, MAX_EPISODE_TIMESTEPS)
    environment = Environment.create(environment =environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize =VISUALIZE)

    #NUM_STATES_AUTOMATON = 4

    #HIDDEN_STATE_SIZE = 64

    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON
    #AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_EXPERTS


    discount_factor = 0.9

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


    #Debugging prints
    print("Istantiated an agent for training with parameters: ")
    print(args)

    print("The goal sequence is: ")
    print(colors)


    trainer = NonMarkovianTrainer(agent,environment,NUM_STATES_AUTOMATON,AUTOMATON_STATE_ENCODING_SIZE,
                                  SINK_ID,num_colors=num_colors
                                  )


    EPISODES = args.episodes

    #Train the agent

    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)


