from typing import Tuple, Dict
from tqdm.auto import tqdm
from tensorforce.agents import Agent
from gym_sapientino_case.env import SapientinoCase
from copy import deepcopy
import numpy as np
from collections import namedtuple
from gym.wrappers.monitoring import video_recorder
from utils import merge_lists, one_hot_encode
from os.path import join
import os
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
                synthetic_exp:bool=False,
                save_path:str='',
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
        self.save_path = save_path

        assert act_pattern in ['act-observe', 'act-experience-update']
        assert not synthetic_exp or (synthetic_exp and act_pattern == 'act-experience-update')

        if DEBUG:
            print("\n################### Agent architecture ###################\n")
            print(self.agent.get_architecture())

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
        Transition = namedtuple('Transition', 's a r s_ i t')
        try:
            # Train for N episodes
            for episode in tqdm(range(episodes),desc='training',leave = True):
                # Record episode experience
                episode_states = list()
                episode_internals = list()
                episode_actions = list()
                episode_terminal = list()
                episode_reward = list()

                transitions = list() # record the whole history

                terminal = False
                # Episode using independent-act and agent.intial_internals()
                internals = agent.initial_internals()
                states = environment.reset()
                prev_states = tuple(states)
                # automaton_state = states['gymtpl1'][0]
                states = self.pack_states(states)
                prevAutState = 0
                # Save the reward that you reach in the episode inside a linked list. 
                # This will be used for nice plots in the report.
                ep_reward = 0.0
                
                while not terminal:
                    environment.render()
                    
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
                    prev_prev_states = tuple(states)
                    # Extract gym sapientino state and the state of the automaton.
                    automaton_state = int(states[1][0])
                    
                    states = self.pack_states(states)
                    # Reward shaping.
                    reward, terminal = self.get_reward(automaton_state, prevAutState, reward, terminal, episode)

                    if self.act_pattern == 'act-experience-update':
                        # act-experience-update
                        episode_terminal.append(terminal)
                        episode_reward.append(reward)
                    
                    if reward > 0:
                        print("Automaton state: {} \t Terminal: {} \t Reward: {} \t Info: {}".format(automaton_state, terminal, reward, info))

                    prevAutState = int(automaton_state)
                    ep_reward += reward
                    cum_reward += reward

                    transitions.append(Transition(prev_states,actions,reward,states,internals, terminal))
                    prev_states = tuple(prev_prev_states)
                    if self.act_pattern == 'act-observe':
                        agent.observe(terminal=terminal, reward=reward)
  
                    if terminal:
                        states = environment.reset()

                print('Episode {}: {}'.format(episode, ep_reward))                

                if self.synthetic:
                    transitions.reverse() # reverse the order to use pop in position -1 (default) -> should not copy the entire list, done only once here
                    # Record synthetic episode experience
                    synthetic_episode_states = list()
                    synthetic_episode_internals = list()
                    synthetic_episode_actions = list()
                    synthetic_episode_terminal = list()
                    synthetic_episode_reward = list()

                    terminal = False
                    # Episode using independent-act and agent.intial_internals()
                    synthetic_internals = agent.initial_internals()
                    synthetic_environment = environment.get_synthetic_env()
                    states = synthetic_environment.reset()

                    # automaton_state = states['gymtpl1'][0]
                    states = self.pack_states(states)
                    prevAutState = 0
                    # Save the reward that you reach in the episode inside a linked list. 
                    # This will be used for nice plots in the report.
                    ep_reward = 0.0
                    while len(transitions):
                        # synthetic_environment.render()
                        
                        transition = transitions.pop()
                        
                        states = transition.s
                        actions = transition.a
                        synthetic_internals = transition.i
                        
                        automaton_state = states[1][0]
                        states = self.pack_states(states)
                        # act-experience-update

                        if len(transitions):
                            for prevAutState in range(0,self.num_state_automaton-2):
                                states_u = states.copy()
                                states_u['gymtpl1'] = one_hot_encode(prevAutState,self.automaton_encoding_size,self.num_state_automaton)
                                states_, reward, terminal, info = synthetic_environment.step(state=prevAutState,action=actions)

                                # Extract gym sapientino state and the state of the automaton.
                                automaton_state = states_[1][0]
                                # states = self.pack_states(states)
                                # Reward shaping.
                                reward, terminal = self.get_reward(automaton_state, prevAutState, reward, terminal, episode)
                                synthetic_episode_states.append(states_u)
                                synthetic_episode_internals.append(synthetic_internals)
                                synthetic_episode_actions.append(actions)
                                synthetic_episode_terminal.append(terminal)
                                synthetic_episode_reward.append(reward)
                        else:
                            reward, terminal = transition.r, transition.t
                            # act-experience-update
                            synthetic_episode_states.append(states)
                            synthetic_episode_internals.append(synthetic_internals)
                            synthetic_episode_actions.append(actions)
                            synthetic_episode_terminal.append(terminal)
                            synthetic_episode_reward.append(reward)
                        

                        prevAutState = int(automaton_state)
                        ep_reward += reward

                        if terminal:
                            states = synthetic_environment.reset()

                    print('Synthetic Episode {}: {}'.format(episode, ep_reward))

                if self.act_pattern == 'act-experience-update':

                    if self.synthetic:
                        episode_states  = merge_lists(episode_states, synthetic_episode_states)
                        episode_actions = merge_lists(episode_actions, synthetic_episode_actions)
                        episode_reward  = merge_lists(episode_reward, synthetic_episode_reward)
                        episode_internals   = merge_lists(episode_internals, synthetic_episode_internals)
                        episode_terminal    = merge_lists(episode_terminal, synthetic_episode_terminal)
                        # episode_states.extend(synthetic_episode_states)
                        # episode_internals.extend(synthetic_episode_internals)
                        # episode_actions.extend(synthetic_episode_actions)
                        # episode_terminal.extend(synthetic_episode_terminal)
                        # episode_reward.extend(synthetic_episode_reward)

                    # Feed recorded experience to agent
                    agent.experience(
                        states=episode_states, internals=episode_internals, actions=episode_actions,
                        terminal=episode_terminal, reward=episode_reward
                    )
                    # Perform update
                    agent.update()
            
            # EVALUATE for 100 episodes and VISUALIZE
            sum_rewards = 0.0
            max_reward  = 0.0
            vid = video_recorder.VideoRecorder(environment,path=join(self.save_path,"video.mp4"))
            temp_mp4 = join(self.save_path,"video_temp.mp4")
            temp_meta_json = join(self.save_path,"video_temp.meta.json")
            for _ in tqdm(range(100), desc='evaluate'):
                states = environment.reset()
                prevAutState = 0
                states = self.pack_states(states)

                internals = agent.initial_internals()
                terminal = False
                while not terminal:
                    environment.render()
                    vid.capture_frame()
                    actions, internals = agent.act(
                        states=states, internals=internals, independent=True, deterministic=True
                    )
                    states, reward, terminal, info = environment.step(action=actions)
                    automaton_state = states[1][0]
                    states = self.pack_states(states)
                    # Reward shaping.
                    reward, terminal = self.get_reward(automaton_state, prevAutState, reward, terminal, episode)
                    prevAutState = automaton_state
                    sum_rewards += reward

                if sum_rewards > max_reward:
                    # Change the filename from video_temp.mp4 to video.mp4
                    vid.path = join(self.save_path,"video.mp4")
                    # Save to local disk only if best reward
                    vid.close()
                    # Create a new temp vid to catch next episode that could have higher reward
                    vid = video_recorder.VideoRecorder(environment,path=temp_mp4)
                else:
                    # If the episode doesn't have higher score, generate a new temp VideoRecorder that overwrite the older file
                    del vid
                    vid = video_recorder.VideoRecorder(environment,path=temp_mp4)

            print('Mean evaluation return:', sum_rewards / 100.0)

            # Remove the temp files
            os.remove(temp_mp4)
            os.remove(temp_meta_json)
            # Close both the agent and the environment.
            agent.close()
            environment.close()
            

            return dict(cumulative_reward_nodiscount = cum_reward,
                        average_reward_nodiscount = cum_reward/episodes)
        finally:
           #Let the user interrupt
           pass

    def get_reward(self, automaton_state, prev_automaton_state, reward, terminal, episode) -> Tuple[float, bool]:
        if self.num_colors == 1:
            if automaton_state == 1 and prev_automaton_state == 0:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True
                
        elif self.num_colors == 2:
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
