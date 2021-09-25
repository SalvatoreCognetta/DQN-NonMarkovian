import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
#from nets import *
from gym_sapientino_case.env import SapientinoCase
import time
from gym.wrappers import TimeLimit

import sys
import os
import configparser
import argparse
import pickle

import importlib



# A function that format the state given by the
# sapientino environment in such a way it is
# good to be putted in the nets
def state2tensor(state):
    modified_state = [state[0][0], state[0][1], state[0][2], state[0][3]]
    for el in state[1]:
        modified_state.append(el)
    state = np.array(modified_state)
    return torch.from_numpy(state).float()



# Class to memorize the history of env exploration
# used to train the nets in batch
class Memory():

    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs, \
                self.values, \
                self.rewards, \
                self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)



# Function to train the nets (actor and critic) in batches
def train(memory, q_val, adam_critic, adam_actor, gamma=0.99):
    
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))

    # Target values are calculated backward
    # it's super important to handle correctly done states,
    # for those caes we want our to target to be equal to the reward only
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + gamma*q_val*(1.0-done)
        q_vals[len(memory)-1 - i] = q_val   # store values from the end to the beginning

    # Compute the advantage
    advantage = torch.Tensor(q_vals) - values
    
    # Compute the critic loss and update
    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()

    # Compute the actor loss and update
    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()



def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train an AC agent on SapientinoCase.')
    parser.add_argument('experiment_dir')
    parser.add_argument('--render_interval', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--evaluate', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    render_interval = args.render_interval
    episodes = args.episodes
    render = args.render
    evaluate = args.evaluate
    resume = args.resume

    experiment_cfg = configparser.ConfigParser()
    experiment_cfg.read(os.path.join(experiment_dir, 'params.cfg'))
    env_cfg = experiment_cfg['ENVIRONMENT']
    agent_cfg = experiment_cfg['AGENT']
    other_cfg = experiment_cfg['OTHER']

    colors = env_cfg['colors'].replace(' ', '').split(',')
    map_file = os.path.join(experiment_dir, env_cfg['map_file'])
    max_episode_timesteps = env_cfg.getint("max_episode_timesteps")
    x0 = env_cfg.getint("initial_x")
    y0 = env_cfg.getint("initial_y")
    
    batch_size = agent_cfg.getint('batch_size')
    lr = agent_cfg.getfloat('learning_rate')
    seed = other_cfg.getint('seed', -1)
    history_file = os.path.join(experiment_dir, "history.pkl")

    if seed >= 0:
        torch.manual_seed(seed)

    # Import the networks from the folder
    if(experiment_dir[-1] == "/"):
        nets_dir = experiment_dir[:-1]
    else:
        nets_dir = experiment_dir
    Actor = getattr(importlib.import_module(nets_dir + ".nets"), "Actor")
    Critic = getattr(importlib.import_module(nets_dir + ".nets"), "Critic")

    env_params = dict(
      reward_per_step=0.0,
      reward_outside_grid=0.0,
      reward_duplicate_beep=0.0,
      acceleration=0.2,
      angular_acceleration=10.0,
      max_velocity=0.4,
      min_velocity=0.0,
      max_angular_vel=40,
      initial_position=[x0, y0],
      tg_reward=1.0,
    )

    # Create the environment
    env = SapientinoCase(
        colors=colors,
        map_file=map_file,
        logdir=experiment_dir,
        params=env_params
    )

    # Set the max number of steps
    env = TimeLimit(env, max_episode_timesteps)

    # Initialize dimensions
    state_dim = 4 + 1
    n_actions = 5
    n_states = env.observation_space[-1].nvec[0]

    # Initialize the nets
    actor = Actor(state_dim, n_actions, n_states)
    critic = Critic(state_dim, n_states)

    # Load the models if resuming
    if resume:
        print('Loading previously saved model weights')
        actor.load_model_weights(os.path.join(experiment_dir, "actor.weights"))
        critic.load_model_weights(os.path.join(experiment_dir, "critic.weights"))
        print('Loading episode history')
        with open(history_file, "rb") as f:
            cum_rewards, steps, total_time = pickle.load(f)
    else:
        cum_rewards = []
        steps = []
        total_time = 0

    cycle_interval = 1000

    for cycles in range(int(episodes/cycle_interval)):
        
        # Take the time in which the cycle starts
        starting_time = time.time()

        # Train the nets
        new_cum_rewards, new_steps = sapientino_training(env, actor, critic, lr, batch_size, cycle_interval, render, render_interval)
        cum_rewards += new_cum_rewards
        steps += new_steps
    
        print()
        print('Saving model weights')
        # Save the models
        actor.save_model_weights(os.path.join(experiment_dir, "actor.weights"))
        critic.save_model_weights(os.path.join(experiment_dir, "critic.weights"))

        # Compute the time so far
        ending_time = time.time()
        total_time += ending_time - starting_time

        print('Saving episode history')
        with open(history_file, "wb") as f:
            pickle.dump((cum_rewards,steps,total_time), f)

        # Print how many episodes done so far and the total reward
        n_episodes_done = cycle_interval * (cycles+1)
        total_rwd = sum(cum_rewards)
        print(f"[ --- Episodes completed so far: [{n_episodes_done}] --- Total reward [{total_rwd}] --- ", end="")
        print("in [{:.5f} s] ---".format(total_time))

        # Eval the model
        if(evaluate):
            eval_rwd = sapientino_eval(actor, critic, env, render, n_episodes=1)
    
            
def sapientino_training(env, actor, critic, lr, batch_size, n_episodes, render, render_interval):

    # Initialize optimizers
    adam_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=lr)

    # Initialize the memory
    memory = Memory()

    # MAIN LOOP
    cum_rewards = []
    steps = []
    step_times = []
    for ep in range(n_episodes):
        done = False
        total_reward = 0
        state = env.reset()
        steps.append(0)

        print(f"\rCurrent train episode [{ep}]", end="")

        cum_rewards.append(0.)
        step_times.append(0)

        while not done:
            
            # Render the environment
            if render and ep % render_interval == render_interval - 1:
                env.render()
                time.sleep(0.01)

            # Sample the action from a Categorical distribution (to allow exploration)
            probs = actor(state2tensor(state))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()

            # Apply the action on the env and take the observation (next_state)
            start_time = time.time()
            next_state, reward, done, info = env.step(action.detach().data.numpy())
            end_time = time.time()

            # Compute the time of a env step
            step_times[ep] = end_time - start_time

            # Update cumulative reward, number of steps and the state
            cum_rewards[ep] += reward
            steps[ep] += 1
            state = next_state
            
            # Add new infos in the memory
            memory.add(dist.log_prob(action), critic(state2tensor(state)), reward, done)

            # Train if done or num steps > batch_size
            if done or (steps[ep] % batch_size == 0):
                last_q_val = critic(state2tensor(next_state)).detach().data.numpy()
                train(memory, last_q_val, adam_critic, adam_actor)
                memory.clear()
            
        # Print the average reward among the last 10 episodes
        if ep % 10 == 9:
            avg_cum_reward = sum(cum_rewards[-10:]) / 10
            print(f' Last 10 episodes avg cum rewards: ', end="")
            print("[" + str(sum(cum_rewards[-10:]) / 10) + "]", end="")
            avg_step_time = sum(step_times[-10:]) / 10
            avg_step_time *= 1000.0
            print(' Step Time Mean: [{:.5f} ms]'.format(avg_step_time), end="")

    return cum_rewards, steps
    

def sapientino_eval(actor, critic, env, render, n_episodes=1):

    # Eval
    print("Evaluation Started.")
    tot_reward = 0.0
    ep = 0
    for ep in range(n_episodes):
        state = env.reset()
        done = False

        print(f"\rCurrent eval episode: {ep}", end="")

        while not done:
            
            env.render()
            time.sleep(0.02)
            
            # Sample the action from a Categorical distribution
            probs = actor(state2tensor(state))
            action = torch.argmax(probs, dim=0)

            # Apply the action on the env and take the observation (next_state)
            next_state, reward, done, info = env.step(action.detach().data.numpy())
            
            # Udpate the state
            state = next_state

            tot_reward += reward
    
    print(f"\nEVAL RESULT: {tot_reward}\n")
    
    return tot_reward



if __name__ == '__main__':
    print("\n")
    main()
