
import numpy as np
from tensorforce.agents import Agent

def build_agent(agent, batch_size,environment,num_states_automaton,
				hidden_layer_size,automaton_state_encoding_size,discount_factor,
				memory= 20000,#'minimum',
				update_frequency = 20,multi_step = 10,exploration = 0.0, learning_rate = 0.001,
                non_markovian = True, entropy_regularization = 0.0, saver = None) ->  Agent:


    """
    Desc: simple function that creates the agent parameters dictionary and manages
    the code to define relevant hyperparameters. It defines also the structure
    of the policy (and the baseline) networks.

    Args:
        agent: (string) the name of the deep reinforcement learning algorithm used to train the agent.
        memory: (int) the size of the agent memory.
        batch_size: (int) the size of experience batch collected by the agent.
        environment: (gym_sapientino_case.env.SapientinoCase) istance of the SapientinoCase environment in which the agent is trained.
        num_states_automaton: (int) number of states of the goal state DFA.
        automaton_state_encoding_size: (int) size of the binary encoding of the automaton state. See the report in report/pdf in section "Non markovian agent" for further details.
        hidden_layer_size: (int) number of neurons of the policy network hidden layer (default implementation features two hidden layers with an equal number of neurons).
        non_markovian: (bool) boolean flag specifying whether or not to istantiate an agent with a non markovian policy network. In the project the markovian agent is used essentially as a baseline for comparisons.
        update_frequency: (int) frequency of updates (default 20).
        multi_step: (int) number of optimization steps, update_frequency * multi_step should be at least 1 if relative subsampling_fraction (default: 10).
        exploration: (float) exploration, defined as the probability for uniformly random output in case of bool and int actions, and the standard deviation of Gaussian noise added to every output in case of float actions, specified globally or per action-type or -name (default: no exploration).
        learning_rate: (float) optimizer learning rate (default: 0.001)
        entropy_regularization: (float) entropy regularization loss weight, to discourage the policy distribution from being “too certain” (default: no entropy regularization).
        saver: (dict)
    Returns:
		Tensorforce Agent
    """

    # Istantiate a default saver if not passed as argument.
    if not saver:
        saver = dict(directory='model')

    AUTOMATON_STATE_ENCODING_SIZE = automaton_state_encoding_size

    # Istantiate a policy network for a non markovian agent.
    # Dictionary containing the agent configuration parameters
    agent = Agent.create(
        agent = 'tensorforce',
        # environment = environment,
        states = dict(
            gymtpl0 = dict(type ='float',shape= (7,),min_value = -float("inf"),max_value = float("inf")), # state space is (x,y,theta,beep)
            gymtpl1 = dict(type ='float',shape=(AUTOMATON_STATE_ENCODING_SIZE,),min_value = 0.0, max_value = 1.0)
        ),
        actions = dict(type = 'int',shape= (1,),num_values=6), # output is 6 action space (left, right, forward, null, beep)
        memory = dict(type="replay",capacity=memory,device="CPU"),
        max_episode_timesteps = 1000,#*(num_states_automaton+1), # needed otherwise the agent will return error, since makes more timesteps than env
        # 
        update = dict(unit="timesteps",batch_size=batch_size,frequency=update_frequency,start=None),
        policy = dict(
            type="parametrized_value_policy",
            network=dict( # The actor network which computes the policy.
                type="custom",
                layers= [
                            dict(type = 'retrieve',tensors= ['gymtpl0']),
                            dict(type = 'linear_normalization'),
                            dict(type='dense', bias = True,activation = 'tanh',size=AUTOMATON_STATE_ENCODING_SIZE),
                            dict(type= 'register',tensor = 'gymtpl0-dense1'),

                            # Perform the product between the one hot encoding of the automaton and the output of the dense layer.
                            dict(type = 'retrieve',tensors=['gymtpl0-dense1','gymtpl1'], aggregation = 'product'),
                            dict(type='dense', bias = True,activation = 'tanh',size=AUTOMATON_STATE_ENCODING_SIZE),
                            dict(type= 'register',tensor = 'gymtpl0-dense2'),
                            dict(type = 'retrieve',tensors=['gymtpl0-dense2','gymtpl1'], aggregation = 'product'),
                            dict(type='register',tensor = 'gymtpl0-embeddings'),
                        ],
                # single_output=True,
                # state_value_mode="implicit",
                # device="CPU",
                # l2_regularization=0.0
            )
        ),
        optimizer = dict(
            optimizer="adam",
            learning_rate=learning_rate,
            gradient_norm_clipping = None,
            clipping_threshold = None,
            multi_step = multi_step,#1
            subsampling_fraction = 1.0,
            linesearch_iterations = 0,
            doublecheck_update = False
        ),

        objective = dict(
            type="action_value",
            huber_loss = None,
            early_reduce = True
        ),
        reward_estimation = dict(
            horizon = 1,
            discount = 0.99,
            predict_horizon_values = "late",
            estimate_advantage = False,
            predict_action_values = False,
            reward_processing = None,
            return_processing = None,
            advantage_processing = None,
            predict_terminal_values = False
        ),

        baseline = dict(
            type="parametrized_value_policy",
            network=dict( # The actor network which computes the policy.
                type="custom",
                layers= [
                            dict(type = 'retrieve',tensors= ['gymtpl0']),
                            dict(type = 'linear_normalization'),
                            dict(type='dense', bias = True,activation = 'tanh',size=AUTOMATON_STATE_ENCODING_SIZE),
                            dict(type= 'register',tensor = 'gymtpl0-dense1'),

                            # Perform the product between the one hot encoding of the automaton and the output of the dense layer.
                            dict(type = 'retrieve',tensors=['gymtpl0-dense1','gymtpl1'], aggregation = 'product'),
                            dict(type='dense', bias = True,activation = 'tanh',size=AUTOMATON_STATE_ENCODING_SIZE),
                            dict(type= 'register',tensor = 'gymtpl0-dense2'),
                            dict(type = 'retrieve',tensors=['gymtpl0-dense2','gymtpl1'], aggregation = 'product'),
                            dict(type='register',tensor = 'gymtpl0-embeddings'),
                        ],
                # single_output=True,
                # state_value_mode="implicit",
                # device="CPU",
                # l2_regularization=0.1
            )
        ),
        baseline_optimizer = dict(
            type="synchronization",
            update_weight = 1.0,
            sync_frequency = 1
        ),
        baseline_objective = None,
        l2_regularization = 0.0,
        state_preprocessing = "linear_normalization",
        variable_noise = 0.0,
        exploration = exploration,
        saver=saver,
        summarizer=dict(directory='summaries',summaries=['reward','graph']),
        entropy_regularization = entropy_regularization
    )


    
   

    
    return agent


