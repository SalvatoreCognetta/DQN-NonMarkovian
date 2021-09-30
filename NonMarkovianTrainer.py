from typing import Tuple, Dict
from tqdm.auto import tqdm
from tensorforce.agents import Agent
from gym_sapientino_case.env import SapientinoCase
from copy import deepcopy
import numpy as np

from utils import one_hot_encode

DEBUG = False

class NonMarkovianTrainer(object):
    def __init__(self,
                agent: Agent,
                environment: SapientinoCase,
                num_state_automaton: int,
                automaton_encoding_size: int,
                sink_id: int, 
                num_colors: int = 2,
                act_pattern:str='act-observe',
                synthetic_exp:bool=False
                ) -> None:

        """
        Desc: class that implements the non markovian training (multiple colors for gym sapientino).
        Keep in mind that the class instantiates the agent according to the parameter dictionary stored inside
        "agent_params" variable. The agent, and in particular the neural network, should be already non markovian.

        Args:
            agent: (tensorforce.agents.Agent) tensorforce agent (algrithm) that will be used to train the policy network (example: ppo, ddqn,dqn).
            environment: (gym_sapientino_case.env.SapientinoCase) istance of the SapientinoCase/openAI gym environment used for training.
            num_state_automaton: (int) number of states of the goal state DFA.
            automaton_encoding_size: (int) size of the binary encoding of the automaton state. See the report in report/pdf in section "Non markovian agent" for further details.
            sink_id: (int) the integer representing the failure state of the goal DFA.
            num_colors: (int) the integer representing the failure state of the goal DFA.
            act_pattern: (str) interaction pattern used int tensorforce, can be act-observe or act-experience-update (not working for tensorforce bug, reward not grow)
            synthetic_exp: (bool) if true feeds the network with synthetic states
        """
        self.num_state_automaton = num_state_automaton
        self.automaton_encoding_size = automaton_encoding_size
        self.sink_id = sink_id
        self.agent = agent
        self.environment = environment
        self.num_colors = num_colors
        self.act_pattern = act_pattern
        self.synthetic = synthetic_exp

        assert act_pattern in ['act-observe', 'act-experience-update']

        if DEBUG:
            print("\n################### Agent architecture ###################\n")
            print(self.agent.get_architecture())

    def make_experience(self, curr_automaton_state, agent, states, environment, episode):
        # experience = []
        for prev_automaton_state in range(self.num_state_automaton):
            if prev_automaton_state!=self.num_state_automaton-1 and prev_automaton_state !=curr_automaton_state:
                states_ = states

                states_['gymtpl1'][0] = prev_automaton_state
                states_ = self.pack_states(tuple(states_.values()))
                actions = agent.act(states=states_)
                
                # deepcopy avoid to increase env steps while iterating over automaton state
                # to be substituted if exists a way to simulate env without increasing timestep and env state
                states_, reward, terminal, info = environment.step(actions)
    
                #Extract gym sapientino state and the state of the automaton.
                automaton_state = int(states_[1][0])
                # Reward shaping.
                reward, terminal = self.get_reward_automaton(automaton_state, prev_automaton_state,  reward, terminal, episode)
                agent.observe(terminal=terminal, reward=reward)            
                # experience.append([prev_states,prev_automaton_state,actions,reward,states,automaton_state])

    def get_reward_automaton(self, automaton_state, prev_automaton_state, reward, terminal, episode) -> Tuple[float, bool]:
        # Terminate the episode with a negative reward
        # if the goal DFA reaches SINK state (failure).
        if automaton_state == self.sink_id:
            reward = -500.0
            terminal = True
            return reward, terminal

        if self.num_colors == 2:
            if automaton_state == 1 and prev_automaton_state == 0:
                reward = 500.0

            elif automaton_state == 3 and prev_automaton_state == 1:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        elif self.num_colors == 3:
            if automaton_state == 1 and prev_automaton_state == 0:
                reward = 500.0

            elif automaton_state == 3 and prev_automaton_state == 1:
                reward = 500.0

            elif automaton_state == 4 and prev_automaton_state == 3:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        elif self.num_colors == 4:
            if automaton_state == 1 and prev_automaton_state == 0:
                reward = 500.0

            elif automaton_state == 3 and prev_automaton_state == 1:
                reward = 500.0

            elif automaton_state == 4 and prev_automaton_state == 3:
                reward = 500.0

            elif automaton_state == 5 and prev_automaton_state == 4:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        return reward, terminal

    def pack_states(self,states) -> Dict[np.ndarray, np.ndarray]:
        """
            Desc: utility function that packs the state dictionary so that it can be passed as input to the
                non markovian agent.

            Args:
                states: (dict) a python dictionary with two keys:
                    'gymtpl0': (np.ndarray) contains the 7 element floating point vector representing the gym sapientino state vector.
                    'gymtpl1': (int) represents the automaton state.

            Returns:
                    python dictionary with two keys:
                        'gymtpl0': (np.ndarray) contains the 7 element floating point vector representing the gym sapientino state vector.
                        'gymtpl1': (np.ndarray) the binary encoded representation for the automaton state (see the pdf report in ./report section "Non markovian agent" for additional details.
        """

        obs = states[0]
        automaton_state = states[1][0]

        # Prepare the encoded automaton state.
        one_hot_encoding = one_hot_encode(automaton_state,
                                            self.automaton_encoding_size,self.num_state_automaton)

        return dict(gymtpl0 = obs,
                    gymtpl1 = one_hot_encoding)


    def train(self, episodes = 1000) -> Dict[float, float]:
        """
            episodes: (int) number of training episodes.
        """

        # The training loop is inspired by the Tensorforce agent "act observe" paradigm 
        # https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html

        cum_reward = 0.0
        agent = self.agent
        environment = self.environment
        try:
            # Train for N episodes
            for episode in tqdm(range(episodes),desc='training',leave = True):
                # Record episode experience
                episode_states = list()
                episode_internals = list()
                episode_actions = list()
                episode_terminal = list()
                episode_reward = list()

                terminal = False
                # Episode using independent-act and agent.intial_internals()
                internals = agent.initial_internals()
                states = environment.reset()
                # automaton_state = states['gymtpl1'][0]
                states = self.pack_states(states)
                prevAutState = 0
                # Save the reward that you reach in the episode inside a linked list. 
                # This will be used for nice plots in the report.
                ep_reward = 0.0
                while not terminal:
                    environment.render()
                    prev_states = states.copy()

                    if self.act_pattern == 'act-observe':
                        actions = agent.act(states=states)
                    elif self.act_pattern == 'act-experience-update':
                        # act-experience-update
                        episode_states.append(states)
                        episode_internals.append(internals)
                        actions, internals = agent.act(states=states, internals=internals, independent=True)
                        # act-experience-update
                        episode_actions.append(actions)
                    
                    states, reward, terminal, info = environment.step(actions)

                    # Extract gym sapientino state and the state of the automaton.
                    automaton_state = states[1][0]
                    states = self.pack_states(states)
                    # Reward shaping.
                    reward, terminal = self.get_reward(automaton_state, prevAutState, reward, terminal, episode)

                    if self.act_pattern == 'act-experience-update':
                        # act-experience-update
                        episode_terminal.append(terminal)
                        episode_reward.append(reward)
                    
                    if reward != -0.1:
                        print("Automaton state: {} \t Terminal: {} \t Reward: {} \t Info: {}".format(automaton_state, terminal, reward, info))

                    prevAutState = int(automaton_state)
                    ep_reward += reward
                    cum_reward += reward
                    
                    if self.act_pattern == 'act-observe':
                        agent.observe(terminal=terminal, reward=reward)
  
                    if terminal:
                        states = environment.reset()
                    # else:
                    #     self.make_experience(automaton_state, agent, prev_states, environment, episode)
                
                # if self.synthetic:
                # while not terminal:
                #     environment.render()
                #     prev_states = states.copy()

                #     if self.act_pattern == 'act-observe':
                #         actions = agent.act(states=states)
                #     elif self.act_pattern == 'act-experience-update':
                #         # act-experience-update
                #         episode_states.append(states)
                #         episode_internals.append(internals)
                #         actions, internals = agent.act(states=states, internals=internals, independent=True)
                #         # act-experience-update
                #         episode_actions.append(actions)
                    
                #     states, reward, terminal, info = environment.step(actions)

                #     # Extract gym sapientino state and the state of the automaton.
                #     automaton_state = states[1][0]
                #     states = self.pack_states(states)
                #     # Reward shaping.
                #     reward, terminal = self.get_reward(automaton_state, prevAutState, reward, terminal, episode)

                #     if self.act_pattern == 'act-experience-update':
                #         # act-experience-update
                #         episode_terminal.append(terminal)
                #         episode_reward.append(reward)
                    
                #     if reward != -0.1:
                #         print("Automaton state: {} \t Terminal: {} \t Reward: {} \t Info: {}".format(automaton_state, terminal, reward, info))

                #     prevAutState = int(automaton_state)
                #     ep_reward += reward
                #     cum_reward += reward
                    
                #     if self.act_pattern == 'act-observe':
                #         agent.observe(terminal=terminal, reward=reward)
  
                #     if terminal:
                #         states = environment.reset()

                print('Episode {}: {}'.format(episode, ep_reward))

                if self.act_pattern == 'act-experience-update':
                    # Feed recorded experience to agent
                    agent.experience(
                        states=episode_states, internals=episode_internals, actions=episode_actions,
                        terminal=episode_terminal, reward=episode_reward
                    )

                    # Perform update
                    agent.update()
                
            # # EVALUATE for 100 episodes and VISUALIZE
            # sum_rewards = 0.0
            # for _ in range(100):
            #     states = environment.reset()
            #     prevAutState = 0
            #     states = self.pack_states(states)
            #     environment.visualize = True
            #     internals = agent.initial_internals()
            #     terminal = False
            #     while not terminal:
            #         actions, internals = agent.act(
            #             states=states, internals=internals, independent=True, deterministic=True
            #         )
            #         states, terminal, reward = environment.execute(actions=actions)
            #         automaton_state = states['gymtpl1'][0]
            #         states = self.pack_states(states)
            #         # Reward shaping.
            #         # reward, terminal = self.get_reward_automaton(automaton_state, prevAutState, reward, terminal, episode)
            #         prevAutState = automaton_state
            #         sum_rewards += reward
            # environment.visualize = False
            # print('Mean evaluation return:', sum_rewards / 100.0)

                


            # Close both the agent and the environment.
            agent.close()
            environment.close()

            return dict(cumulative_reward_nodiscount = cum_reward,
                        average_reward_nodiscount = cum_reward/episodes)
        finally:
           #Let the user interrupt
           pass

    def get_reward(self, automaton_state, prev_automaton_state, reward, terminal, episode) -> Tuple[float, bool]:
        if self.num_colors == 2:
            if automaton_state == 1 and prev_automaton_state == 0:
                reward = 500.0

            elif automaton_state == 2 and prev_automaton_state == 1:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        elif self.num_colors == 3:
            if automaton_state == 1 and prev_automaton_state == 0:
                reward = 500.0

            elif automaton_state == 2 and prev_automaton_state == 1:
                reward = 500.0

            elif automaton_state == 3 and prev_automaton_state == 2:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        elif self.num_colors == 4:
            # Terminate the episode with a negative reward if the goal DFA reaches SINK state (failure).
            
            if automaton_state == 1 and prev_automaton_state == 0:
                reward = 500.0

            elif automaton_state == 2 and prev_automaton_state == 1:
                reward = 500.0

            elif automaton_state == 3 and prev_automaton_state == 2:
                reward = 500.0

            elif automaton_state == 4 and prev_automaton_state == 3:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        return reward, terminal
