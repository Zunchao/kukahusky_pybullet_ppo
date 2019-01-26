# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example configurations using the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

from kukahusky_pybullet_ppo.agent import ppo
from kukahusky_pybullet_ppo.agent import networks
import tensorflow as tf
#import kukahusky_pybullet_ppo.env.mmKukaHuskyGymEnv as mmKukaHuskyGymEnv
#import kukahusky_pybullet_ppo.env.neobotixschunkGymEnv as neobotixschunkGymEnv

from kukahusky_pybullet_ppo.env.mmKukaHuskyGymEnv import MMKukaHuskyGymEnv
from kukahusky_pybullet_ppo.env.neobotixschunkGymEnv import NeobotixSchunkGymEnv

def default():
  """Default configuration for PPO."""
  # General
  # help(mmKukaHuskyGymEnv)
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30
  use_gpu = False
  # Network
  network = networks.feed_forward_gaussian
  weight_summaries = dict(
      all=r'.*',
      policy=r'.*/policy/.*',
      value=r'.*/value/.*')
  policy_layers = 128, 128
  value_layers = 128, 128
  init_mean_factor = 0.1
  init_logstd = -1
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  update_epochs_policy = 64
  update_epochs_value = 64
  learning_rate = 3*1e-4
  # Losses
  discount = 0.995
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  return locals()


def pybullet_pendulum():
  locals().update(default())
  env = 'InvertedPendulumBulletEnv-v0'
  max_length = 200
  steps = 5e7  # 50M
  return locals()

def pybullet_doublependulum():
  locals().update(default())
  env = 'InvertedDoublePendulumBulletEnv-v0'
  max_length = 1000
  steps = 5e7  # 50M
  return locals()

def pybullet_pendulumswingup():
  locals().update(default())
  env = 'InvertedPendulumSwingupBulletEnv-v0'
  max_length = 1000
  steps = 5e7  # 50M
  return locals()

def pybullet_cheetah():
  """Configuration for MuJoCo's half cheetah task."""
  locals().update(default())
  # Environment
  env = 'HalfCheetahBulletEnv-v0'
  max_length = 1000
  steps = 1e8  # 100M
  return locals()

def pybullet_ant():
  locals().update(default())
  env = 'AntBulletEnv-v0'
  max_length = 1000
  steps = 5e7  # 50M
  return locals()

def pybullet_kuka_grasping():
  """Configuration for Bullet Kuka grasping task."""
  locals().update(default())
  # Environment
  env = 'KukaBulletEnv-v0'
  max_length = 1000
  steps = 1e7  # 10M
  return locals()


def pybullet_racecar():
  """Configuration for Bullet MIT Racecar task."""
  locals().update(default())
  # Environment
  env = 'RacecarBulletEnv-v0' #functools.partial(racecarGymEnv.RacecarGymEnv, isDiscrete=False, renders=True)
  max_length = 10
  steps = 1e7  # 10M
  return locals()

def pybullet_neoschunk_reaching():
  """Configuration for Bullet Kukahusky mm task."""
  locals().update(default())
  env = functools.partial(NeobotixSchunkGymEnv, isDiscrete=False, renders=False, action_dim = 9, rewardtype='rdense')
  # Environment
  max_length = 1000
  steps = 2e8  # 100M
  return locals()

def pybullet_kukahusky_reaching():
  """Configuration for Bullet Kukahusky mm task."""
  locals().update(default())
  env = functools.partial(MMKukaHuskyGymEnv, isDiscrete=False, renders=False, action_dim = 9, rewardtype='rdense')
  # Environment
  max_length = 1000
  steps = 2e8  # 100M
  return locals()