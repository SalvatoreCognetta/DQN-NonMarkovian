from tensorforce.agents import Agent
from tensorforce.environments import Environment


from one_hot import one_hot_encode
from tqdm.auto import tqdm
from copy import deepcopy



DEBUG = False


class NonMarkovianTrainer(object):
    def __init__(self,agent,environment,num_state_automaton,
                 automaton_encoding_size,sink_id, num_colors = 2,
                 ):

        """
        Desc: class that implements the non markovian training (multiple colors for gym sapientino).
        Keep in mind that the class instantiates the agent according to the parameter dictionary stored inside
        "agent_params" variable. The agent, and in particular the neural network, should be already non markovian.

        Args:
            @param agent: (tensorforce.agents.Agent) tensorforce agent (algrithm) that will be used to train the policy network (example: ppo, ddqn,dqn).
            @param environment: (tensorforce.environments.Environment) istance of the tensorforce/openAI gym environment used for training.
            @param num_state_automaton: (int) number of states of the goal state DFA.
            @automaton_state_encoding_size: (int) size of the binary encoding of the automaton state. See the report in report/pdf in section "Non markovian agent" for further details.
            @sink_id: (int) the integer representing the failure state of the goal DFA.

        """



        self.num_state_automaton = num_state_automaton
        self.automaton_encoding_size = automaton_encoding_size

        self.sink_id = sink_id


        #Create both the agent and the environment that will be used a training time.
        self.agent = agent
        self.environment = environment


        self.num_colors = num_colors



        if DEBUG:
            print("\n################### Agent architecture ###################\n")
            architecture = self.agent.get_architecture()
            print(architecture)


    def make_experience(self, agent, states, environment, episode):
        # experience = []

        for prev_automaton_state in range(self.num_state_automaton):

            states_ = states.copy()
            states_['gymtpl1'][0] = prev_automaton_state
            states_ = self.pack_states(states_)
            actions = agent.act(states=states_)
            
            # deepcopy avoid to increase env steps while iterating over automaton state
            # to be substituted if exists a way to simulate env without increasing timestep and env state
            states_, terminal, reward = deepcopy(environment).execute(actions=actions)
 
            #Extract gym sapientino state and the state of the automaton.
            automaton_state = states_['gymtpl1'][0]
            # Reward shaping.
            reward, terminal = self.get_reward_automaton(automaton_state, prev_automaton_state,  reward, terminal, episode)
            agent.observe(terminal=terminal, reward=reward)            
            # experience.append([prev_states,prev_automaton_state,actions,reward,states,automaton_state])

    def get_reward_automaton(self, automaton_state, prev_automaton_state, reward, terminal, episode):
        if self.num_colors == 2:

            if automaton_state == self.sink_id:
                reward = -500.0
                terminal = True

            elif automaton_state == 1 and prev_automaton_state==0:
                reward = 500.0

            elif automaton_state == 3 and prev_automaton_state==1:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        elif self.num_colors == 3:

            if automaton_state == self.sink_id:
                reward = -500.0

                terminal = True

            elif automaton_state == 1 and prev_automaton_state==0:
                reward = 500.0

            elif automaton_state == 3 and prev_automaton_state==1:
                reward = 500.0

            elif automaton_state ==4 and prev_automaton_state == 3:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        elif self.num_colors == 4:

            #Terminate the episode with a negative reward if the goal DFA reaches SINK state (failure).
            if automaton_state == self.sink_id:
                reward = -500.0

                terminal = True


            elif automaton_state == 1 and prev_automaton_state==0:
                reward = 500.0

            elif automaton_state == 3 and prev_automaton_state==1:
                reward = 500.0

            elif automaton_state ==4 and prev_automaton_state == 3:
                reward = 500.0

            elif automaton_state == 5:
                reward = 500.0
                print("Visited goal on episode: ", episode)
                terminal = True

        return reward, terminal


    def pack_states(self,states):
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


        obs = states['gymtpl0']
        automaton_state = states['gymtpl1'][0]


        """
            Prepare the encoded automaton state.
        """
        one_hot_encoding = one_hot_encode(automaton_state,
                                            self.automaton_encoding_size,self.num_state_automaton)

        return dict(gymtpl0 =obs,
                    gymtpl1 = one_hot_encoding)

    def train(self,episodes = 1000):

        """
            @param episodes: (int) number of training episodes.
        """
        cum_reward = 0.0



        

        agent = self.agent
        environment = self.environment



        """
            The training loop is inspired by the Tensorforce agent "act observe" paradigm 
            https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html
        """



        try:

            # Train for N episodes
            for episode in tqdm(range(episodes),desc='training',leave = True):
                terminal = False
                # Episode using independent-act and agent.intial_internals()
                states = environment.reset()
                # automaton_state = states['gymtpl1'][0]
                states = self.pack_states(states)
                prevAutState = 0
                #Save the reward that you reach in the episode inside a linked list. This will be used for nice plots in the report.
                ep_reward = 0.0
                while not terminal:
                    prev_states = states
                    actions = agent.act(states=states)
                    states, terminal, reward = environment.execute(actions=actions)
                    #Extract gym sapientino state and the state of the automaton.
                    automaton_state = states['gymtpl1'][0]
                    states = self.pack_states(states)
                    # Reward shaping.
                    reward, terminal = self.get_reward_automaton(automaton_state, prevAutState, reward, terminal, episode)
                    prevAutState = automaton_state
                    ep_reward += reward
                    cum_reward += reward
                    agent.observe(terminal=terminal, reward=reward)
  
                    if terminal:
                        states = environment.reset()
                    else:
                        self.make_experience(agent, prev_states, environment, episode)
                
                print('Episode {}: {}'.format(episode, ep_reward))
                
                # EVALUATE for 100 episodes and VISUALIZE
                # sum_rewards = 0.0
                # for _ in range(10):
                #     states = environment.reset()
                #     prevAutState = 0
                #     states = self.pack_states(states)
                #     environment.visualize = True
                #     internals = agent.initial_internals()
                #     terminal = False
                #     while not terminal:
                #         prev_states = states
                #         actions, internals = agent.act(
                #             states=states, internals=internals, independent=True, deterministic=True
                #         )
                #         states, terminal, reward = environment.execute(actions=actions)
                #         automaton_state = states['gymtpl1'][0]
                #         states = self.pack_states(states)
                #         # Reward shaping.
                #         reward, terminal = self.get_reward_automaton(automaton_state, prevAutState, reward, terminal, episode)
                #         prevAutState = automaton_state
                #         sum_rewards += reward
                # environment.visualize = False
                # print('Mean evaluation return:', sum_rewards / 100.0)

                


            #Close both the agent and the environment.
            agent.close()
            environment.close()


            return dict(cumulative_reward_nodiscount = cum_reward,
                        average_reward_nodiscount = cum_reward/episodes)
        finally:

           #Let the user interrupt
           pass




