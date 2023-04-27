####################################################################################
#
# 10/4/23 DH: Based on https://www.tensorflow.org/agents/tutorials/9_c51_tutorial
#
####################################################################################

# 10/4/23 DH: Refactor of tf_agents test suite
import tensorflow as tf
from tf_agents.networks import categorical_q_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.utils import common

from suite_gym_utils import *
from gym_config import *

# ------------------------------------------------------------------------------------------

num_atoms = 51
fc_layer_params = (100,)

learning_rate = 1e-3

min_q_value = -20
max_q_value = 20
gamma = 0.99

num_eval_episodes = 10

# 16/4/23 DH: Refactoring Google obfuscation...
#collect_steps_per_iteration = 1

log_interval = 200
eval_interval = 1000

#num_iterations = 15000
num_iterations = 2000

# ---------------------------------------------------------------

# 10/4/23 DH:
initGym()

# ==================================== TF-Agents ======================================

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    gym_config.train_env.observation_spec(),
    gym_config.train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

# 8/4/23 DH: https://github.com/tensorflow/agents/blob/c3bc54423efb68e69d6ecfdb2ae259595da76d74/tf_agents/agents/categorical_dqn/categorical_dqn_agent.py#L225
"""
if boltzmann_temperature is not None:
  ...
else:
  collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
      policy, epsilon=self._epsilon_greedy)

policy = greedy_policy.GreedyPolicy(policy)
"""

agent = categorical_dqn_agent.CategoricalDqnAgent(
    gym_config.train_env.time_step_spec(),
    gym_config.train_env.action_spec(),

    categorical_q_network=categorical_q_net,
    optimizer=optimizer,

    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)

#print("\nPRE agent.initialize()")
#print("agent.collect_policy: ",type(agent.collect_policy))
#print("agent.collect_policy:",agent.collect_policy.collect_data_spec,"\n")

agent.initialize()

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)
# =====================================================================================

initReplayBuffer(agent, gym_config.train_env)

# --------------- Interface Gym with CategoricalDqnAgent training ---------------------

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1. (from initial random policy action)
dataset = gym_config.replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)

# =================== TRAIN ====================
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(gym_config.eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# 9/4/23 DH:
dirNum = None

# 8/4/23 DH: num_iterations = 15000
for iterNum in range(num_iterations):

  # 8/4/23 DH: https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/CategoricalDqnAgent#attributes

  # 8/4/23 DH: EpsilonGreedyPolicy (see above for CategoricalDqnAgent github code)
  #
  # Add trajectory to the replay buffer (which gets accessed via dataset+iterator)
  collect_step(gym_config.train_env, agent.collect_policy)

  ########################################################################################################
  # 12/4/23 DH:
  # 1 - Put next action Trajectory (from state + policy) of Train Env in Replay Buffer
  # 2 - Get it out of Replay Buffer
  # 3 - Train the agent based on Trajectory:
  #     {'step_type', 'observation', 'action', 'policy_info', 'next_step_type', 'reward', 'discount'}
  #
  #     https://github.com/tensorflow/agents/blob/master/tf_agents/agents/dqn/dqn_agent.py#L391
  #     https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py#L621
  #
  # [Repeat for specified iterations]
  ########################################################################################################

  # Sample a batch of data from the buffer (prev from random policy) and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience)

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    #replayBufSize = tf.get_static_value(replay_buffer.num_frames())
    #print("TRAIN replay_buffer:",replayBufSize )
    #print("agent.train_step_counter:",step,", iteration:",iterNum+1)

  if step % eval_interval == 0:
    # 8/4/23 DH: num_eval_episodes = 10
    avg_return = compute_avg_return(gym_config.eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    returns.append(avg_return)
    
    # 9/4/23 DH:
    filename = 'imageio'+str(step)+'.mp4'
    dirNum = createEpisodeVideo(agent, gym_path, filename, dirNum=dirNum)

# ------------------- DISPLAY RESULTS --------------------

createReturnsGraph(returns, "avg-return.jpg", gym_path, dirNum=dirNum)

createEpisodeVideo(agent, gym_path, gym_filename, dirNum=dirNum)

#filepath = os.path.join(path, filename)
#os.system("open " + filepath)

"""
C51 tends to do slightly better than DQN on CartPole-v1, but the difference between the two agents becomes more and more significant in increasingly complex environments. For example, on the full Atari 2600 benchmark, C51 demonstrates a mean score improvement of 126% over DQN after normalizing with respect to a random agent. Additional improvements can be gained by including n-step updates.

For a deeper dive into the C51 algorithm, see [A Distributional Perspective on Reinforcement Learning (2017)](https://arxiv.org/pdf/1707.06887.pdf).
"""
