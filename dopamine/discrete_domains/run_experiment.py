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
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.hadqn import hadqn_agent
from dopamine.agents.hadqn import hra_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.agents.hr_cvae import hr_cvae_agent
from dopamine.agents.hrmmd import hrmmd_agent
from dopamine.agents.mmdqn import mmdqn_agent
from dopamine.agents.mmdqn_nd import mmdqn_nd_agent
from dopamine.agents.mmdqn_nd import mmdqn_nd_ev_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
from dopamine.discrete_domains.reward_logger_update import reward_logger, MultiRewardEnv
from dopamine.discrete_domains.distribution_logger import distribution_logger
from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent as jax_implicit_quantile_agent
from dopamine.jax.agents.quantile import quantile_agent as jax_quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent as jax_rainbow_agent

import numpy as np
import tensorflow as tf

import gin.tf


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.compat.v1.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  debug_mode = True  # TODO: maybe change back
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'hadqn':
    return hadqn_agent.HADQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'hra':
    return hra_agent.HRAAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'hr_cvae':
    return hr_cvae_agent.HrCVAEAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'hrmmd':
    return hrmmd_agent.HRMMDAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'mmdqn':
    return mmdqn_agent.MMDAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'mmdqn_nd':
    return mmdqn_nd_agent.MMDAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'mmdqn_nd_ev':
    return mmdqn_nd_ev_agent.MMDAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_dqn':
    return jax_dqn_agent.JaxDQNAgent(num_actions=environment.action_space.n,
                                     summary_writer=summary_writer)
  elif agent_name == 'jax_quantile':
    return jax_quantile_agent.JaxQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_rainbow':
    return jax_rainbow_agent.JaxRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_implicit_quantile':
    return jax_implicit_quantile_agent.JaxImplicitQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval', **kwargs):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent, **kwargs)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent, **kwargs)
  elif schedule == 'eval':
    return EvalRunner(base_dir, create_agent, **kwargs)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class Runner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               clip_rewards=True,
               prerun=False,
               monte_carlo_samples=200,
               reward_logdir='./reward',
               exp_name='MD3QN-initial-721'):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.compat.v1.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    self.exp_name = exp_name
    print(f'=== prerun: {prerun} ===')
    print(f'=== clip_rewards: {clip_rewards} ===')
    print(f'=== monte_carlo_samples: {monte_carlo_samples} ===')
    print(f'=== exp_name: {exp_name} ===')
    assert base_dir is not None
    tf.compat.v1.disable_v2_behavior()
    if prerun:
      pass
    else:
      if 'MultiRewardMaze' not in reward_logdir:
        create_environment_fn = lambda: MultiRewardEnv(atari_lib.create_atari_environment())
        reward_logger.load(reward_logdir)
      else:
        create_environment_fn = atari_lib.create_atari_environment
    distribution_logger.set_logdir(base_dir)

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._clip_rewards = clip_rewards
    self._create_directories()
    self._summary_writer = tf.compat.v1.summary.FileWriter(self._base_dir)
    self.monte_carlo_samples = monte_carlo_samples

    self._environment = create_environment_fn()

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.
    self._sess = tf.compat.v1.Session('', config=config)
    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=self._summary_writer)
    if 'evaluation_setting' in dir(self._agent) and self._agent.evaluation_setting:
      self.evaluation_setting = True
    else:
      self.evaluation_setting = False

    self._summary_writer.add_graph(graph=tf.compat.v1.get_default_graph())
    self._sess.run(tf.compat.v1.global_variables_initializer())

    if self.evaluation_setting:
      self._environment_test = create_environment_fn()
      self._agent.eval_policy_load_fn()

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
    self.prerun = prerun
    if prerun:
      # self._training_steps = 25000
      self._evaluation_steps = 0
      self.reward_logdir = f'{reward_logdir}-{num_iterations}'

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix,
                                                   checkpoint_frequency=1)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    print(f'latest_checkpoint_version: {latest_checkpoint_version}')
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        logging.info('Reloaded checkpoint and will start from iteration %d',
                     self._start_iteration)

  def _initialize_episode(self, **kwargs):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation, **kwargs)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward, **kwargs):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
    """
    self._agent.end_episode(reward, **kwargs)

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    constrain_env = ('constrain_env' in dir(self._environment) and self._environment.constrain_env)

    step_number = 0
    total_reward = 0.

    if constrain_env:
      is_less_constraint = [c == '<' for c in self._environment.constrain_type]
      initial_constraint = np.array(self._environment.constrain_value)
      # print(f'is_less_constraint: {is_less_constraint}')
      # print(f'initial_constraint at initial step 0: {initial_constraint}')
      initial_kwargs = {'remained_constraint': initial_constraint,
                        'is_less_constraint': is_less_constraint}
      remained_constraint = np.array(self._environment.constrain_value)
    else:
      initial_kwargs = {}

    action = self._initialize_episode(**initial_kwargs)
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)
      if self.prerun:
        reward_logger.log(reward)  # TODO: remove debugging outputs here
      elif type(reward) is np.ndarray:
        primitive_reward = np.sum(reward)
      else:
        reward, primitive_reward = reward

      if self.prerun:
        total_reward += reward
      else:
        total_reward += primitive_reward
      step_number += 1

      if constrain_env:
        remained_constraint = (remained_constraint - np.array(reward)) / self._agent.gamma
        # print(f'remained_constraint at step {step_number}: {remained_constraint}')
        kwargs = {'remained_constraint': remained_constraint,
                  'is_less_constraint': is_less_constraint}
      else:
        kwargs = {}

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._agent.end_episode(reward, **kwargs)
        action = self._agent.begin_episode(observation, **initial_kwargs)
        if constrain_env:
          remained_constraint = initial_constraint
          # print('is_terminal !!')
      else:
        action = self._agent.step(reward, observation, **kwargs)

    self._end_episode(reward, **kwargs)
    if constrain_env:
      satisfy_constraint = [(0 > rc) ^ l for rc, l in zip(remained_constraint, is_less_constraint)]
      all_satisfy_constraint = all(satisfy_constraint)
      # print(f'finished episode, remained_constraint is {remained_constraint}')
      # print(f'all_satisfy_constraint: {all_satisfy_constraint}')
      if all_satisfy_constraint:
        total_reward = 1.0
      else:
        total_reward = 0.0


    if self.prerun:
      # print('==== reward logging ====')
      print(reward_logger)
      # print('==== reward logging ====')

    return step_number, total_reward

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    evaluate_in_this_iteration = (run_mode_str == 'train') and (distribution_logger.iteration % 5 == 0)
    if self._agent.maze_env:
      evaluate_in_this_iteration = (run_mode_str == 'train')
    eval_after_steps = 100
    if self._agent.eval_policy is None:
      # eval_after_steps > 0 is not supported for the maze case where "eval_policy" is None
      eval_after_steps = 0

    if self.evaluation_setting and evaluate_in_this_iteration:
      if eval_after_steps == 0:
        obs = self._environment_test.reset()
        mc_samples = self._environment_test.monte_carlo_joint_return(
          obs, num_samples=self.monte_carlo_samples, initial_action=0, gamma=self._agent.gamma,
          eval_policy=self._agent.eval_policy, trained_agent=self._agent
        )
      else:
        obs = self._environment_test.reset()
        for _ in range(eval_after_steps):
          if _ == 0:
            action = self._agent.eval_policy.begin_episode(obs)
          else:
            action = self._agent.eval_policy.step(reward, obs)
          obs, reward, terminate, _ = self._environment_test.step(action)
          reward = np.array(reward[0])
        initial_action = self._agent.eval_policy.step(reward, obs)
        mc_samples = self._environment_test.monte_carlo_joint_return_middle(
          num_samples=self.monte_carlo_samples, initial_action=initial_action, gamma=self._agent.gamma,
          eval_policy=self._agent.eval_policy, trained_agent=self._agent
        )
        eval_state = self._agent.eval_policy.state

    while step_count < min_steps:
      if self._agent.maze_env:
        log_per_episodes = 500
      else:
        log_per_episodes = 10
      if self.evaluation_setting and evaluate_in_this_iteration and (num_episodes % log_per_episodes == 0):
        if eval_after_steps == 0:
          particles = self._agent.joint_value_distribution(obs, 0)
        else:
          particles = self._agent.joint_value_distribution_stacked(eval_state, initial_action)

        print(f'prediction: mean={particles.mean()}, std={particles.std()}, max={particles.max()}, min={particles.min()}')
        print(f'mc_samples: mean={mc_samples.mean()}, std={mc_samples.std()}, max={mc_samples.max()}, min={mc_samples.min()}')
        total_train_steps = step_count + distribution_logger.iteration * self._training_steps
        distribution_logger.log_evaluation(particles, mc_samples, num_episodes,
                                           num_steps=total_train_steps)

      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      if self.evaluation_setting and num_episodes % 500 == 0:
        print(f'{num_episodes}-th episode, length {episode_length}, total reward: {episode_return}')

      # We use sys.stdout.write instead of logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per second.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    average_steps_per_second = number_steps / time_delta
    statistics.append(
        {'train_average_steps_per_second': average_steps_per_second})
    logging.info('Average undiscounted return per training episode: %.2f',
                 average_return)
    logging.info('Average training steps per second: %.2f',
                 average_steps_per_second)
    return num_episodes, average_return, average_steps_per_second

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train, num_episodes_eval,
                                     average_reward_eval,
                                     average_steps_per_second)
    if os.path.isfile('azcopy'):
      local_log_dir = self._base_dir
      bash_command = "sudo ./azcopy cp " + local_log_dir + f" \"https://msradrlstorage.blob.core.windows.net/mycontainer/pushi/MD3QN/{self.exp_name}/?sv=2019-02-02&ss=bfqt&srt=sco&sp=rwdlacup&se=2022-04-16T17:46:39Z&st=2020-04-16T09:46:39Z&spr=https&sig=GJrYBLnmdSayaEdKqFMxhDpzVAIxctchW59tkQVw0mY%3D\" --recursive=true"

      response = os.popen(bash_command).readlines()
      print("Copy data to azure", response)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval,
                                  average_steps_per_second):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
      average_steps_per_second: float, The average number of steps per second.
    """
    summary = tf.compat.v1.Summary(value=[
        tf.compat.v1.Summary.Value(
            tag='Train/NumEpisodes', simple_value=num_episodes_train),
        tf.compat.v1.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward_train),
        tf.compat.v1.Summary.Value(
            tag='Train/AverageStepsPerSecond',
            simple_value=average_steps_per_second),
        tf.compat.v1.Summary.Value(
            tag='Eval/NumEpisodes', simple_value=num_episodes_eval),
        tf.compat.v1.Summary.Value(
            tag='Eval/AverageReturns', simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      distribution_logger.set_iteration(iteration)
      statistics = self._run_one_iteration(iteration)
      if not self.prerun:
        self._log_experiment(iteration, statistics)
        self._checkpoint_experiment(iteration)
      else:
        reward_logger.export(self.reward_logdir + '.pkl')
        with open(f'{self.reward_logdir}.txt', 'w') as f:
          f.write(reward_logger.__repr__())
        print(reward_logger)
        print('==== reward log finished ====')
    self._summary_writer.flush()


class EvalRunner(Runner):
  def run_experiment(self):
    statistics = self._run_one_iteration(0)

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
      statistics)

    self._save_tensorboard_summaries(iteration, 0,
                                     0, num_episodes_eval,
                                     average_reward_eval,
                                     0)
    return statistics.data_lists


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train,
                                     average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, average_steps_per_second):
    """Save statistics as tensorboard summaries."""
    summary = tf.compat.v1.Summary(value=[
        tf.compat.v1.Summary.Value(
            tag='Train/NumEpisodes', simple_value=num_episodes),
        tf.compat.v1.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward),
        tf.compat.v1.Summary.Value(
            tag='Train/AverageStepsPerSecond',
            simple_value=average_steps_per_second),
    ])
    self._summary_writer.add_summary(summary, iteration)
