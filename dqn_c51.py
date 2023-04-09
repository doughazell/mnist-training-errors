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

# Set up a virtual display for rendering OpenAI gym environments.
# 4/4/23 DH: "pyvirtualdisplay.abstractdisplay.XStartError: No success after 10 retries."
#display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

"""
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
of the FIXED FORCE the cart is pushed with.

| Num | Action                 |
|-----|------------------------|
| 0   | Push cart to the left  |
| 1   | Push cart to the right |

**Note**: 
The VELOCITY that is reduced or increased by the applied force IS NOT FIXED and it depends on the angle
the pole is pointing. The CENTER OF GRAVITY OF THE POLE varies the amount of energy needed to move the 
cart underneath it

| Num | Observation           | Min                 | Max               |
|-----|-----------------------|---------------------|-------------------|
| 0   | Cart Position         | -4.8                | 4.8               |
| 1   | Cart Velocity         | -Inf                | Inf               |
| 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3   | Pole Angular Velocity | -Inf                | Inf               |

"""

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

# 9/4/23 DH:
filename = 'imageio.mp4'
path = 'video'

# ---------------------------------------------------------------
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    # 8/4/23 DH:
    iCnt = 0
    while not time_step.is_last():
      iCnt += 1
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    print(_,"-",iCnt)
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

# 8/4/23 DH: https://github.com/tensorflow/agents/blob/master/tf_agents/policies/epsilon_greedy_policy.py
def collect_step(environment, policy):
  #print("collect_step():",type(policy))

  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)

  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

# 7/4/23 DH:
def addGraphic(imgEdit,cartVel):
  # ->, -->, --->, ---->, ----->, =====>
  # + vel = ">"
  # - vel = "<"
  # Number of prepended '-' based on vel
  # Cart speed:= 0 -> 1
  #              0, 0.2, 0.4, 0.6, 0.8,  1
  #                  ->  -->  ---> ----> ----->
  
  cartSpeed = abs(cartVel)

  # https://pandas.pydata.org/docs/reference/api/pandas.cut.html
  #pd.cut(df, bins=[0,10,20,100], labels=[10,50,80])
  #pd.to_numeric(pd.cut(df['A'], bins=[0,10,20,100], labels=[10,50,80]))

  quant = pd.cut([cartSpeed], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1,2,3,4,5])
  dashNum = quant[0]

  dashes = ""
  if math.isnan(dashNum):
    # 8/4/23 DH: Outside specified 'bins' range
    dashes = "====="
  else:
    dashes = ('-' * dashNum)

  arrowStr = ""
  if cartVel > 0:
    arrowStr = dashes + ">"
    
  else:
    arrowStr = "<" + dashes

  imgEdit.text((300,370), arrowStr, (33, 32, 30))

def createReturnsGraph(returns):
  steps = range(0, num_iterations + 1, eval_interval)
  plt.plot(steps, returns)
  plt.ylabel('Average Return')
  plt.xlabel('Step')
  plt.ylim(top=550)

  # 8/4/23 DH: 'plt.show() needs to follow 'plt.savefig()' to prevent saved image being blank
  #            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
  plt.savefig("avg-return.jpg")
  
  plt.show()

"""
def embed_mp4(filename):
  #Embeds an mp4 file in the notebook.
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)
"""

def createEpisodeVideo(path,filename,dirNum=None):
  img = None
  iCnt = 0
  
  # 9/4/23 DH: Check for 'video' + add if not found
  if not os.path.exists(path):
    os.makedirs(path)

  if dirNum == None:
    dirNum = 1
    dirpath = os.path.join(path, str(dirNum))
    while os.path.exists(dirpath):
      dirNum += 1
      dirpath = os.path.join(path, str(dirNum))
    os.makedirs(dirpath)
  else:
    dirpath = os.path.join(path, str(dirNum))

  filepath = os.path.join(dirpath, filename)

  with imageio.get_writer(filepath, fps=1) as video:
    # 5/4/23 DH:
    print("video:",video)

    # 9/4/23 DH: 1 episode of trained agent from suite_gym reset()
    num_episodes = 1
    # 5/4/23 DH: https://docs.python.org/3/reference/lexical_analysis.html#reserved-classes-of-identifiers
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())

      iCnt = 0
      while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)

        # 6/4/23 DH: https://github.com/tensorflow/agents/blob/master/tf_agents/trajectories/time_step.py
        obs = time_step.observation.numpy()[0]
        cartPos = obs[0]
        cartVel = obs[1]
        poleAng = obs[2]
        poleVel = obs[3]
        
        #print("Cart Position: %8.5f, Cart Velocity: %8.5f, Pole Angle: %8.5f, Pole Velocity: %8.5f" 
        #      % (cartPos, cartVel, poleAng, poleVel))
        #print("Cart Velocity: %4.1f" % (cartVel))

        # 5/4/23 DH:
        iCnt += 1
        img = eval_py_env.render()

        imgXtra = Image.fromarray(img)

        # 6/4/23 DH: https://github.com/tensorflow/agents/blob/master/tf_agents/trajectories/trajectory.py
        #            'action_step' is an 'tf_agents.trajectories.policy_step'
        if action_step.action.numpy()[0] == 0:
          actionStr = "Left"
        else:
          actionStr = "Right"

        imgEdit = ImageDraw.Draw(imgXtra)
        imgEdit.text((300,350), actionStr, (33, 32, 30))

        # 7/4/23 DH: Add arrow with size + direction based on 'cartVel' USING ASCII-ART...obvs...
        #            https://www.appsloveworld.com/coding/python3x/69/how-can-i-draw-an-arrow-using-pil
        # ->, -->, --->, ---->, ----->, =====>
        addGraphic(imgEdit,cartVel)

        video.append_data( numpy.array(imgXtra) )

  # 5/4/23 DH: 'img' now contains last frame added to video
  #print("\nLast:",_, ","+str(iCnt), type(img), img.shape)
  #with imageio.get_writer('lastFrame.jpg') as imgFile:
  #  imgFile.append_data(img)

  return dirNum

# ---------------------------------------------------------------

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# ======================= TF-Agents ========================
categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
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
    train_env.time_step_spec(),
    train_env.action_spec(),

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

# ==========================================================

# ----------------- Runtime Cfg --------------------

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

#compute_avg_return(eval_env, random_policy, num_eval_episodes)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

replayBufSize = tf.get_static_value(replay_buffer.num_frames())
print("\nINIT replay_buffer:",replayBufSize )
# 8/4/23 DH: 'initial_collect_steps = 1000'
# 9/4/23 DH: needs to be > 1 
#           "[TFUniformReplayBuffer is empty. Make sure to add items before sampling the buffer.]"
initial_collect_steps = 2
for _ in range(initial_collect_steps):
  # Add trajectory to the replay buffer (which gets accessed via dataset)
  collect_step(train_env, random_policy)

replayBufSize = tf.get_static_value(replay_buffer.num_frames())
print("PRIMED replay_buffer:",replayBufSize,"\n" )
# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1. (from random policy action)
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)
# --------------------------------------------------

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# 9/4/23 DH:
dirNum = None

# =================== TRAIN ====================
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
    collect_step(train_env, agent.collect_policy)

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
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    returns.append(avg_return)
    
    # 9/4/23 DH:
    dirNum = createEpisodeVideo(path,'imageio'+str(step)+'.mp4',dirNum=dirNum)

# ------------------- DISPLAY RESULTS --------------------

createReturnsGraph(returns)

createEpisodeVideo(path,filename, dirNum=dirNum)

#filepath = os.path.join(path, filename)
#os.system("open " + filepath)

"""
C51 tends to do slightly better than DQN on CartPole-v1, but the difference between the two agents becomes more and more significant in increasingly complex environments. For example, on the full Atari 2600 benchmark, C51 demonstrates a mean score improvement of 126% over DQN after normalizing with respect to a random agent. Additional improvements can be gained by including n-step updates.

For a deeper dive into the C51 algorithm, see [A Distributional Perspective on Reinforcement Learning (2017)](https://arxiv.org/pdf/1707.06887.pdf).
"""
