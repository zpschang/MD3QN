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

HRMMDNetworkType = collections.namedtuple(
  'hrmmd_network', ['q_values', 'q_samples']
)

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

class HRMMDNetworkV21(tf.keras.Model):

  def __init__(self, num_actions, reward_dim,
               num_samples=100, name=None):
    super(HRMMDNetworkV21, self).__init__(name=name)

    self.num_actions = num_actions
    self.reward_dim = reward_dim
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
      num_actions * reward_dim * num_samples,
      kernel_initializer=self.kernel_initializer,
      name='fully_connected')

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)  # [batch_size, num_actions * num_samples * reward_dim]

    q_samples = tf.reshape(x, [-1, self.num_actions, self.num_samples, self.reward_dim])
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
    x = self.dense2(x)  # [batch_size, num_actions * num_samples * reward_dim]
    q_samples = tf.reshape(x, [-1, self.num_actions, self.num_samples, self.reward_dim])

    q_samples = tf.reduce_sum(
      q_samples * tf.expand_dims(tf.expand_dims(action_one_hot, axis=-1), axis=-1),
      axis=1
    )  # [batch_size, num_samples, reward_dim]

    return q_samples

class HRMMDNetworkV22(tf.keras.Model):

  def __init__(self, num_actions, reward_dim, quantile_embedding_dim=64,
               num_samples=100, name=None):
    super(HRMMDNetworkV22, self).__init__(name=name)

    self.quantile_embedding_dim = quantile_embedding_dim

    self.num_actions = num_actions
    self.reward_dim = reward_dim
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
      num_actions * reward_dim, activation=activation_fn,
      kernel_initializer=self.kernel_initializer,
      name='fully_connected')

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)

    state_vector_length = x.get_shape().as_list()[-1]
    batch_size = state.get_shape().as_list()[0]

    state_net_tiled = tf.tile(x, [self.num_samples, 1])
    quantiles_shape = [self.num_samples * batch_size, self.reward_dim]
    quantiles = tf.random.uniform(
      quantiles_shape, minval=0, maxval=1, dtype=tf.float32)
    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    range_tensor = tf.tile(tf.range(1, self.quantile_embedding_dim + 1, 1), [self.reward_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(range_tensor, tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    # Create the quantile layer in the first call. This is because
    # number of output units depends on the input shape. Therefore, we can only
    # create the layer during the first forward call, not during `.__init__()`.
    if not hasattr(self, 'dense_quantile'):
      self.dense_quantile = tf.keras.layers.Dense(
        state_vector_length, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer)
    quantile_net = self.dense_quantile(quantile_net)
    x = tf.multiply(state_net_tiled, quantile_net)

    x = self.dense1(x)
    x = self.dense2(x)  # [batch_size * num_samples, num_actions * reward_dim]
    q_samples = tf.reshape(x, [-1, self.num_samples, self.num_actions, self.reward_dim])
    q_samples = tf.transpose(q_samples, [0, 2, 1, 3])  # [batch_size, num_actions, num_samples, reward_dim]

    q_values = tf.reduce_sum(tf.reduce_mean(q_samples, axis=2), axis=2)  # [batch_size, num_actions]

    return HRMMDNetworkType(q_values, q_samples)

  def generate_samples(self, state, action_one_hot):
    q_samples = self.call(state).q_samples

    q_samples = tf.reduce_sum(
      q_samples * tf.expand_dims(tf.expand_dims(action_one_hot, axis=-1), axis=-1),
      axis=1
    )  # [batch_size, num_samples, reward_dim]

    return q_samples