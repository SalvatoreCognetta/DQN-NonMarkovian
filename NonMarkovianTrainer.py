from tensorforce.agents import Agent
from tensorforce.environments import Environment


from one_hot import one_hot_encode
from tqdm.auto import tqdm




DEBUG = False




class NonMarkovianTrainer(object):
    def __init__(self,agent,environment,number_of_experts,
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



        self.number_of_experts = number_of_experts
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





    def train(self,episodes = 1000):

        """
            @param episodes: (int) number of training episodes.
        """





        cum_reward = 0.0



        def pack_states(states):
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
                                              self.automaton_encoding_size,self.number_of_experts)

            return dict(gymtpl0 =obs,
                        gymtpl1 = one_hot_encoding)

        agent = self.agent
        environment = self.environment



        """
            The training loop is inspired by the Tensorforce agent "act observe" paradigm 
            https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html
        """



        try:
            for episode in tqdm(range(episodes),desc='training',leave = True):
                terminal = False

                states = environment.reset()
                
                automaton_state = states['gymtpl1'][0]
                states = pack_states(states)


                prevAutState = 0
                #Save the reward that you reach in the episode inside a linked list. This will be used for nice plots in the report.
                ep_reward = 0.0

                while not terminal:


                    actions = agent.act(states=states)


                    states, terminal, reward = environment.execute(actions=actions)

                    #Extract gym sapientino state and the state of the automaton.
                    automaton_state = states['gymtpl1'][0]
                    states = pack_states(states)


                    """
                        Reward shaping.
                    """

                    if self.num_colors == 2:


                        if automaton_state == self.sink_id:
                            reward = -500.0
                            terminal = True


                        elif automaton_state == 1 and prevAutState==0:
                            reward = 500.0

                        elif automaton_state == 3 and prevAutState==1:
                            reward = 500.0
                            print("Visited goal on episode: ", episode)
                            terminal = True

                    elif self.num_colors == 3:

                        if automaton_state == self.sink_id:
                            reward = -500.0

                            terminal = True


                        elif automaton_state == 1 and prevAutState==0:
                            reward = 500.0

                        elif automaton_state == 3 and prevAutState==1:
                            reward = 500.0

                        elif automaton_state ==4 and prevAutState == 3:
                            reward = 500.0
                            print("Visited goal on episode: ", episode)
                            terminal = True



                    elif self.num_colors == 4:

                        #Terminate the episode with a negative reward if the goal DFA reaches SINK state (failure).
                        if automaton_state == self.sink_id:
                            reward = -500.0

                            terminal = True


                        elif automaton_state == 1 and prevAutState==0:
                            reward = 500.0

                        elif automaton_state == 3 and prevAutState==1:
                            reward = 500.0

                        elif automaton_state ==4 and prevAutState == 3:
                            reward = 500.0

                        elif automaton_state == 5:
                            reward = 500.0
                            print("Visited goal on episode: ", episode)
                            terminal = True



                    prevAutState = automaton_state



                    #Update the cumulative reward during the training.
                    cum_reward+=reward

                    #Update the episode reward during the training
                    ep_reward += reward



                    agent.observe(terminal=terminal, reward=reward)
                    if terminal == True:
                        states = environment.reset()



            #Close both the agent and the environment.
            self.agent.close()
            self.environment.close()


            return dict(cumulative_reward_nodiscount = cum_reward,
                        average_reward_nodiscount = cum_reward/episodes)
        finally:

           #Let the user interrupt
           pass




