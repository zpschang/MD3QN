# coding=utf-8
# Copyright 2018 The Dopamine Authors.
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
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.

## Networks
We are subclassing keras.models.Model in our network definitions. Each network
class has two main functions: `.__init__` and `.call`. When we create our
network the `__init__` function is called and necessary layers are defined. Once
we create our network, we can create the output operations by doing `call`s to
our network with different inputs. At each call, the same parameters will be
used.

More information about keras.Model API can be found here:
https://www.tensorflow.org/api_docs/python/tf/keras/models/Model

## Network Types
Network types are namedtuples that define the output signature of the networks
used. Please use the appropriate signature as needed when defining new networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from absl import logging

import cv2
import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf
from dopamine.discrete_domains.reward_logger_update import reward_logger


NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])
RainbowNetworkType = collections.namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])
CVAENetworkType = collections.namedtuple(
  'cvae_network', ['q_values', 'q_samples', 'conditions']
)
ImplicitQuantileNetworkType = collections.namedtuple(
    'iqn_network', ['quantile_values', 'quantiles'])
HADQNNetworkType = collections.namedtuple('hadqn_network', ['q_values', 'sub_q_values'])
HRMMDNetworkType = collections.namedtuple(
  'hrmmd_network', ['q_values', 'q_samples']
)



@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=True):
  """Wraps an Atari 2600 Gym environment with some basic preprocessing.

  This preprocessing matches the guidelines proposed in Machado et al. (2017),
  "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
  Problems for General Agents".

  The created environment is the Gym wrapper around the Arcade Learning
  Environment.

  The main choice available to the user is whether to use sticky actions or not.
  Sticky actions, as prescribed by Machado et al., cause actions to persist
  with some probability (0.25) when a new command is sent to the ALE. This
  can be viewed as introducing a mild form of stochasticity in the environment.
  We use them by default.

  Args:
    game_name: str, the name of the Atari 2600 domain.
    sticky_actions: bool, whether to use sticky_actions as per Machado et al.

  Returns:
    An Atari 2600 environment with some standard preprocessing.
  """
  assert game_name is not None
  game_version = 'v0' if sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
  if 'MultiRewardMaze' in game_name:
    full_game_name = game_name
  env = gym.make(full_game_name)
  if 'MultiRewardMaze' in game_name:
    reward_logger.rewards = np.zeros(env.reward_dim)
    return env
  # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
  # handle this time limit internally instead, which lets us cap at 108k frames
  # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
  # restoring states.
  env = env.env
  env = AtariPreprocessing(env)
  return env


@gin.configurable(blacklist=['variables'])
def maybe_transform_variable_names(variables, legacy_checkpoint_load=False):
  """Maps old variable names to the new ones.

  The resulting dictionary can be passed to the tf.compat.v1.train.Saver to load
  legacy checkpoints into Keras models.

  Args:
    variables: list, of all variables to be transformed.
    legacy_checkpoint_load: bool, if True the variable names are mapped to
        the legacy names as appeared in `tf.slim` based agents. Use this if
        you want to load checkpoints saved before tf.keras.Model upgrade.
  Returns:
    dict or None, of <new_names, var>.
  """
  logging.info('legacy_checkpoint_load: %s', legacy_checkpoint_load)
  if legacy_checkpoint_load:
    name_map = {}
    for var in variables:
      new_name = var.op.name.replace('bias', 'biases')
      new_name = new_name.replace('kernel', 'weights')
      name_map[new_name] = var
  else:
    name_map = None
  return name_map


class NatureDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(NatureDQNNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)

    return DQNNetworkType(self.dense2(x))

class HADQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, reward_dim, name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(HADQNNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    self.reward_dim = reward_dim
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(num_actions * reward_dim, name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    x = tf.reshape(x, [-1, self.num_actions, self.reward_dim])
    q_values = tf.reduce_sum(x, axis=2)

    return HADQNNetworkType(q_values, x)

# TODO: now fixing activation position
class HRMMDNetwork(tf.keras.Model):

  def __init__(self, num_actions, reward_dim,
               latent_dim=10, hidden_dim=20, condition_dim=40, num_layers=4,
               num_samples=50, name=None):
    super(HRMMDNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    self.reward_dim = reward_dim
    self.latent_dim = latent_dim
    self.hidden_dim = hidden_dim
    self.condition_dim = condition_dim
    self.num_samples = num_samples
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
      scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
      32, [8, 8], strides=4, padding='same', activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
      64, [4, 4], strides=2, padding='same', activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
      64, [3, 3], strides=1, padding='same', activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
      512, activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
      num_actions * condition_dim, activation=activation_fn,
      kernel_initializer=self.kernel_initializer,
      name='fully_connected')

    self.decoder_layers = [
      tf.keras.layers.Dense(
        self.hidden_dim if idx < num_layers - 1 else self.reward_dim,
        activation=activation_fn if idx < num_layers - 1 else None,
        kernel_initializer=self.kernel_initializer, name='decoder_layer'
      ) for idx in range(num_layers)
    ]

  def decode(self, zc):
    for decoder_layer in self.decoder_layers:
      zc = decoder_layer(zc)
    return zc

  def sample(self, c, num_samples=50):
    assert len(c.shape) == 2  # [batch_size * num_actions, condition_dim]
    batch_size, condition_dim = tf.shape(c)[0], tf.shape(c)[1]
    z = tf.random.normal([batch_size, num_samples, self.latent_dim])
    c = tf.tile(tf.expand_dims(c, axis=1), [1, num_samples, 1])
    zc = tf.concat([z, c], axis=-1)
    q_samples = self.decode(zc)  # [batch_size * num_actions, num_samples, reward_dim]
    return q_samples

  def call(self, state, num_samples=None):
    if num_samples is None:
      num_samples = self.num_samples
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)  # [batch_size, num_actions * condition_dim]

    c = tf.reshape(x, [-1, self.num_actions, self.condition_dim])  # [batch_size, num_actions, condition_dim]
    c = tf.reshape(c, [-1, self.condition_dim])  # [batch_size * num_actions, condition_dim]
    q_samples = self.sample(c, num_samples)  # [batch_size * num_actions, num_samples, reward_dim]
    assert len(q_samples.shape) == 3
    q_samples = tf.reshape(q_samples, [-1, self.num_actions, num_samples, self.reward_dim])
    q_values = tf.reduce_sum(tf.reduce_mean(q_samples, axis=2), axis=2)  # [batch_size, num_actions]

    return HRMMDNetworkType(q_values, q_samples)

  def generate_samples(self, state, action_one_hot, num_samples=50):
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)  # [batch_size, num_actions * condition_dim]

    c = tf.reshape(x, [-1, self.num_actions, self.condition_dim])  # [batch_size, num_actions, condition_dim]
    c = tf.reduce_sum(
      c * tf.expand_dims(action_one_hot, axis=-1),
      axis=1
    )  # [batch_size, condition_dim]

    q_samples = self.sample(c, num_samples)  # [batch_size, num_samples, reward_dim]
    return q_samples

class RainbowNetwork(tf.keras.Model):
  """The convolutional network used to compute agent's return distributions."""

  def __init__(self, num_actions, num_atoms, support, name=None):
    """Creates the layers used calculating return distributions.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      name: str, used to crete scope for network parameters.
    """
    super(RainbowNetwork, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)

# TODO: modify network architecture
# TODO: now activation position is not correct!
class CVAENetwork(tf.keras.Model):
  def __init__(self, num_actions, reward_dim, kl_weight,
               latent_dim=10, condition_dim=15, num_layers=4, name=None):
    super(CVAENetwork, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.kl_weight = kl_weight
    self.num_actions = num_actions
    self.condition_dim = condition_dim
    self.reward_dim = reward_dim
    self.latent_dim = latent_dim # TODO: tune hidden dim
    self.hidden_dim = 2 * condition_dim  # TODO: tune hidden dim
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions * condition_dim, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

    self.encoder_layers = [
      tf.keras.layers.Dense(
        self.hidden_dim, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='encoder_layer'
      ) for idx in range(num_layers - 1)
    ]

    self.mu = tf.keras.layers.Dense(
        self.latent_dim, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='encoder_mu_layer'
      )

    self.logvar = tf.keras.layers.Dense(
      self.latent_dim, activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='encoder_logvar_layer'
    )

    self.decoder_layers = [
      tf.keras.layers.Dense(
        self.hidden_dim if idx < num_layers - 1 else self.reward_dim, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='decoder_layer'
      ) for idx in range(num_layers)
    ]

  def encode(self, xc):
    for encoder_layer in self.encoder_layers:
      xc = encoder_layer(xc)
    mu, logvar = self.mu(xc), self.logvar(xc)
    return mu, logvar

  def decode(self, zc):
    for decoder_layer in self.decoder_layers:
      zc = decoder_layer(zc)
    return zc

  def reparameterize(self, mu, logvar):
    std = tf.exp(0.5 * logvar)
    r = tf.random.normal(tf.shape(mu))
    return r * std + mu

  def loss_function(self, q_samples, q_samples_hat, mu, logvar):
    mse_loss = tf.reduce_mean((q_samples - q_samples_hat) ** 2, axis=[1, 2])
    kl_loss = -0.5 * (1 + logvar - mu ** 2 - tf.exp(logvar))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=2), axis=1)

    loss = mse_loss + self.kl_weight * kl_loss
    return {'loss': loss,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss}

  def compute_loss(self, c, q_samples):
    assert len(c.shape) == 2  # [batch_size, condition_dim]
    assert len(q_samples.shape) == 3  # [batch_size, num_samples, reward_dim]
    num_samples = tf.shape(q_samples)[1]
    c = tf.tile(tf.expand_dims(c, axis=1), [1, num_samples, 1])
    xc = tf.concat([q_samples, c], axis=-1)
    mu, logvar = self.encode(xc)  # [batch_size, num_samples, latent_dim]

    z = self.reparameterize(mu, logvar)  # [batch_size, num_samples, latent_dim]
    assert len(z.shape) == 3
    zc = tf.concat([z, c], axis=-1)
    q_samples_hat = self.decode(zc)  # [batch_size, num_samples, reward_dim]

    return self.loss_function(q_samples, q_samples_hat, mu, logvar)

  def sample(self, c, num_samples=50):
    assert len(c.shape) == 2  # [batch_size, condition_dim]
    shape = tf.shape(c)
    batch_size, condition_dim = shape[0], shape[1]
    z = tf.random.normal([batch_size, num_samples, self.latent_dim])
    c = tf.tile(tf.expand_dims(c, axis=1), [1, num_samples, 1])
    zc = tf.concat([z, c], axis=-1)
    q_samples = self.decode(zc)  # [batch_size, num_samples, reward_dim]
    return q_samples

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    conditions = tf.reshape(x, [-1, self.num_actions, self.condition_dim])  # [batch_size, num_actions, condition_dim]
    c = tf.reshape(conditions, [-1, self.condition_dim])  # [batch_size * num_actions, condition_dim]
    num_samples = 50
    q_samples = self.sample(c, num_samples)  # [batch_size * num_actions, num_samples, reward_dim]
    assert len(q_samples.shape) == 3
    q_samples = tf.reshape(q_samples, [-1, self.num_actions, num_samples, self.reward_dim])
    q_values = tf.reduce_mean(q_samples, axis=2)  # [batch_size, num_actions, reward_dim]

    return CVAENetworkType(q_values, q_samples, conditions)

class ImplicitQuantileNetwork(tf.keras.Model):
  """The Implicit Quantile Network (Dabney et al., 2018).."""

  def __init__(self, num_actions, quantile_embedding_dim, name=None):
    """Creates the layers used calculating quantile values.

    Args:
      num_actions: int, number of actions.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      name: str, used to create scope for network parameters.
    """
    super(ImplicitQuantileNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.quantile_embedding_dim = quantile_embedding_dim
    # We need the activation function during `call`, therefore set the field.
    self.activation_fn = tf.keras.activations.relu
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

  def call(self, state, num_quantiles):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: `tf.Tensor`, contains the agent's current state.
      num_quantiles: int, number of quantile inputs.
    Returns:
      collections.namedtuple, that contains (quantile_values, quantiles).
    """
    batch_size = state.get_shape().as_list()[0]
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    state_vector_length = x.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles * batch_size, 1]
    quantiles = tf.random.uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32)
    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
        1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    # Create the quantile layer in the first call. This is because
    # number of output units depends on the input shape. Therefore, we can only
    # create the layer during the first forward call, not during `.__init__()`.
    if not hasattr(self, 'dense_quantile'):
      self.dense_quantile = tf.keras.layers.Dense(
          state_vector_length, activation=self.activation_fn,
          kernel_initializer=self.kernel_initializer, name='dense')
    quantile_net = self.dense_quantile(quantile_net)
    x = tf.multiply(state_net_tiled, quantile_net)
    x = self.dense1(x)
    quantile_values = self.dense2(x)
    return ImplicitQuantileNetworkType(quantile_values, quantiles)


@gin.configurable
class AtariPreprocessing(object):
  """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
               screen_size=84):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))

    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size

    obs_dims = self.environment.observation_space
    # Stores temporary observations used for pooling over two successive
    # frames.
    self.screen_buffer = [
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
    ]

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
               dtype=np.uint8)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    self.environment.reset()
    self.lives = self.environment.ale.lives()
    self._fetch_grayscale_observation(self.screen_buffer[0])
    self.screen_buffer[1].fill(0)
    return self._pool_and_resize()

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    accumulated_reward = 0.

    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # grayscale image from the ALE. This is a little faster.
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward

      if self.terminal_on_life_loss:
        new_lives = self.environment.ale.lives()
        is_terminal = game_over or new_lives < self.lives
        self.lives = new_lives
      else:
        is_terminal = game_over

      if is_terminal:
        break
      # We max-pool over the last two frames, in grayscale.
      elif time_step >= self.frame_skip - 2:
        t = time_step - (self.frame_skip - 2)
        self._fetch_grayscale_observation(self.screen_buffer[t])

    # Pool the last two observations.
    observation = self._pool_and_resize()

    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info

  def _fetch_grayscale_observation(self, output):
    """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    self.environment.ale.getScreenGrayscale(output)
    return output

  def _pool_and_resize(self):
    """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    if self.frame_skip > 1:
      np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                 out=self.screen_buffer[0])

    transformed_image = cv2.resize(self.screen_buffer[0],
                                   (self.screen_size, self.screen_size),
                                   interpolation=cv2.INTER_AREA)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)
