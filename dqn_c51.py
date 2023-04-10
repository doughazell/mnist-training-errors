####################################################################################
#
# 10/4/23 DH: Based on https://www.tensorflow.org/agents/tutorials/9_c51_tutorial
#
####################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import imageio
#import IPython
import matplotlib
import matplotlib.pyplot as plt

#import PIL.Image
# 5/4/23 DH: 
from PIL import Image, ImageFont, ImageDraw 
import numpy
import pandas as pd
import math

import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# 4/4/23 DH: To run mpeg image
import os

# 10/4/23 DH: Refactor of tf_agents test suite
from suite_gym_utils import *
from gym_config import *
import time

# ------------------------------------------------------------------------------------------

# 8/4/23 DH: https://gymnasium.farama.org/environments/classic_control/cart_pole/#rewards
#            "The threshold for rewards is 475 for v1."
env_name = "CartPole-v1" # @param {type:"string"}

#num_iterations = 15000 # @param {type:"integer"}
num_iterations = 2000

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
#replay_buffer_capacity = 100000  # @param {type:"integer"}
# 9/4/23 DH:
replay_buffer_capacity = 3

fc_layer_params = (100,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# ---------------------------------------------------------------
# 10/4/23 DH: TODO: Add these to 'gym_config.py'
"""
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
"""
initGym()

# ======================= TF-Agents ========================
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

# ==========================================================
agent.initialize()

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

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

  # Collect a few steps using collect_policy and save to the replay buffer.
  # 8/4/23 DH: 'collect_steps_per_iteration = 1'...so for, "a few steps", read one...!
  #
  # 8/4/23 DH: https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/CategoricalDqnAgent#attributes
  for _ in range(collect_steps_per_iteration):
    # 8/4/23 DH: EpsilonGreedyPolicy (see above for CategoricalDqnAgent github code)
    #
    # Add trajectory to the replay buffer (which gets accessed via dataset+iterator)
    collect_step(gym_config.train_env, agent.collect_policy)

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
