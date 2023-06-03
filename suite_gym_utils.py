#####################################################
#
# 10/4/23 DH: Refactor of tf_agents test suite
#
#####################################################

import tensorflow as tf

from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import suite_gym
from tf_agents.policies import random_tf_policy
from tf_agents.environments import tf_py_environment

from PIL import Image, ImageFont, ImageDraw
import imageio
import matplotlib.pyplot as plt

import numpy
import pandas as pd
import math
import os

import gym_config

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

# -------------------------------------------------------------------------------------------------
# 10/4/23 DH: Should these be added to a namespace (rather than add namespace to filename)?
# 3/6/23 DH: ...yup we've just had a global variable issue with:
#            'num_iterations' + 'DQNc51.num_iterations' clash...!!!
gym_filename = 'imageio.mp4'
gym_path = 'video'

# 8/4/23 DH: https://gymnasium.farama.org/environments/classic_control/cart_pole/#rewards
#            "The threshold for rewards is 475 for v1."
env_name = "CartPole-v1"

# 3/6/23 DH: Now 'createReturnsGraph()' arg to prevent global variable value clash
#eval_interval = 1000
#num_iterations = 2000

batch_size = 64
n_step_update = 2
replay_buffer_capacity = 3

# ------------------------------------ FUNCTIONS -----------------------------------------------------
def initGym():
  gym_config.train_py_env = suite_gym.load(env_name)
  gym_config.eval_py_env = suite_gym.load(env_name)

  gym_config.train_env = tf_py_environment.TFPyEnvironment(gym_config.train_py_env)
  gym_config.eval_env = tf_py_environment.TFPyEnvironment(gym_config.eval_py_env)

# 11/4/23 DH: https://www.tensorflow.org/agents/api_docs/python/tf_agents/replay_buffers/ReverbReplayBuffer#some_additional_notes
#             ReverbReplayBuffer (from '1_dqn_tutorial.ipynb') vs TFUniformReplayBuffer
#
#             https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers
#             https://github.com/deepmind/reverb :
# "an experience replay system for distributed reinforcement learning algorithms"
# "Reverb currently only supports Linux based OSes."
def initReplayBuffer(agent, train_env):
  random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

  # 10/4/23 DH: Module 'global' variables were still not being set prior to access from another file
  # https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

  gym_config.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_capacity)

  replayBufSize = tf.get_static_value(gym_config.replay_buffer.num_frames())
  print("\nINIT replay_buffer:",replayBufSize )

  # 9/4/23 DH: needs to be > 1 
  #           "[TFUniformReplayBuffer is empty. Make sure to add items before sampling the buffer.]"
  initial_collect_steps = 2
  for _ in range(initial_collect_steps):
    # Add trajectory to the replay buffer (which gets accessed via dataset)
    collect_step(train_env, random_policy)

  replayBufSize = tf.get_static_value(gym_config.replay_buffer.num_frames())
  print("PRIMED replay_buffer:",replayBufSize,"\n" )

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

    print("Episode",_,"=",iCnt,"steps and last step returns:",episode_return.numpy()[0])
    
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
  # 12/4/23 DH: https://github.com/tensorflow/agents/blob/master/tf_agents/trajectories/trajectory.py
  gym_config.replay_buffer.add_batch(traj)

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

def createReturnsGraph(returns, iterations, eval_interval, filename,path,dirNum=None):
  # 10/4/23 DH:
  filepath = os.path.join(path, str(dirNum), filename)

  steps = range(0, iterations + 1, eval_interval)
  plt.plot(steps, returns)
  plt.ylabel('Average Return')
  plt.xlabel('Step')
  plt.ylim(top=550)

  # 8/4/23 DH: 'plt.show() needs to follow 'plt.savefig()' to prevent saved image being blank
  #            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
  plt.savefig(filepath)
  
  plt.show()

def createEpisodeVideo(agent, path, filename, dirNum=None):
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
      time_step = gym_config.eval_env.reset()
      video.append_data(gym_config.eval_py_env.render())

      iCnt = 0
      while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = gym_config.eval_env.step(action_step.action)

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
        img = gym_config.eval_py_env.render()

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

# ------------------------------------ END: FUNCTIONS ------------------------------------------------



