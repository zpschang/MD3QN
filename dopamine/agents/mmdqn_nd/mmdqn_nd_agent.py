import sys, os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import random
from scipy.stats import norm, bernoulli
import collections
from copy import deepcopy

from dopamine.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer, circular_replay_buffer
from dopamine.discrete_domains.particle_net import NdParticleDQNet
from dopamine.discrete_domains import networks_aligned
from dopamine.discrete_domains.distribution_logger import distribution_logger
import gin.tf

ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))


def beta_identity_fn(training_steps, init_value=1):
    return init_value


def beta_intrinsic_suppress_fn(training_steps, init_value=50):
    return init_value * np.sqrt(np.log(training_steps + 3.) / (training_steps + 3.))


@gin.configurable
def bandwidth_linear_annealing_fn(training_steps, init_value=10., final_value=1., annealing_period=1000000):
    if training_steps >= annealing_period:
        return final_value
    else:
        return (training_steps / annealing_period) * final_value + (1. - training_steps / annealing_period) * init_value


def gaussian_rbf_kernel(d, sigmas, kscales):
    """
    Args:
        d: (b,n,n,K) squared distance.
        sigmas: a list of sigma values.
    """
    d_mean = tf.reduce_mean(d, axis=-1)  # (b,n,n)
    tf_sigmas = tf.constant(sigmas, dtype=tf.float32)
    tf_kernel_scales = tf.constant(kscales, dtype=tf.float32)  # (k)
    h = 1. / tf.reshape(tf_sigmas, [-1, 1])  # (k,1)
    s = tf.matmul(h, tf.reshape(d_mean, [1, -1]))  # (k,b*n*n)
    mat = tf.reshape(tf.reduce_sum(tf.exp(-s) * tf.reshape(tf_kernel_scales, [-1, 1]), axis=0), shape=tf.shape(d_mean))  # (b,n,n)
    return mat  # (b,n,n)


def hubber_loss(u, kappa=1):
    if kappa == 0:
        return tf.abs(u)
    else:
        huber_loss_case_one = tf.to_float(tf.abs(u) <= kappa) * 0.5 * u ** 2
        huber_loss_case_two = tf.to_float(tf.abs(u) > kappa) * kappa * (tf.abs(u) - 0.5 * kappa)
        return huber_loss_case_one + huber_loss_case_two


@gin.configurable
class MMDAgent(dqn_agent.DQNAgent):
    """MMDAgent inherited from DQN agent.
    """

    def __init__(self,
                 sess,
                 num_actions=4,
                 num_atoms=10,
                 delta=0.1,
                 kappa=0,
                 target_estimator='mean',
                 policy='eps_greedy',
                 beta_fn=beta_intrinsic_suppress_fn,
                 bandwidth_annealing_fn=bandwidth_linear_annealing_fn,
                 bandwidth_selection_type='med',
                 debug=False,
                 double_dqn=False,
                 observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
                 stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
                 network=NdParticleDQNet,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 monitor_step=10000,
                 epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 replay_scheme='prioritized',
                 tf_device='/cpu:*',
                 use_staging=True,
                 optimizer=tf.train.AdamOptimizer(
                     learning_rate=0.00025, epsilon=0.0003125
                 ),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 network_type='v21',
                 bandwidth_type='v0',
                 kscale_type='v0',
                 use_priority=False,
                 evaluation_setting=False,
                 maze_env=False,
                 eval_policy_path=None,
                 eval_policy_epsilon=0.02,
                 constrain_env=False,
                 use_marginal=False):  # TODO: turn back to 0.02
        """
        Args:
            bandwidth_selection_type: str, [med/annealing/mixture/const]

        """

        if network_type == 'v21':
            network = networks_aligned.NdParticleDQNet
        elif network_type == 'v22':
            network = networks_aligned.HRMMDNetworkV22
        elif network_type == 'v1':
            network = networks_aligned.HRMMDNetwork
        elif network_type == 'v12':
            network = networks_aligned.HRMMDNetworkV12
        elif network_type == 'v3-drdrl':
            network = networks_aligned.NdParticleDQNetDRDRL

        self.bandwidth_type = bandwidth_type
        self.kscale_type = kscale_type
        self.use_priority = use_priority
        self.evaluation_setting = evaluation_setting
        self.maze_env = maze_env
        self.constrain_env = constrain_env
        if maze_env:
            observation_shape = (84, 84, 3)
            stack_size = 1
        print(f'==== network_type: {network_type} ====')
        print(f'==== num_atoms: {num_atoms} ====')
        print(f'==== bandwidth_type: {bandwidth_type} ====')
        print(f'==== kscale_type: {kscale_type} ====')
        print(f'==== use_priority: {use_priority} ====')
        print(f'==== evaluation_setting: {evaluation_setting} ====')
        print(f'==== maze_env: {maze_env} ====')
        print(f'==== constrain_env: {constrain_env} ===')
        print(f'==== eval_policy_path: {eval_policy_path} ====')
        print(f'==== eval_policy_epsilon: {eval_policy_epsilon} ====')
        print(network)

        self.eval_policy_epsilon = eval_policy_epsilon
        self.eval_policy = None
        if eval_policy_path is not None:
            if 'IQN' in eval_policy_path:
                from dopamine.agents.implicit_quantile.implicit_quantile_agent import ImplicitQuantileAgent
                from dopamine.discrete_domains import checkpointer

                self.eval_policy = ImplicitQuantileAgent(sess, num_actions=num_actions)
                self.eval_policy.eval_mode = True
                self.eval_policy.epsilon_eval = eval_policy_epsilon
                checkpoint_dir = os.path.join(eval_policy_path, 'checkpoints')
                _checkpointer = checkpointer.Checkpointer(checkpoint_dir, 'ckpt')
                latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(checkpoint_dir)
                experiment_data = _checkpointer.load_checkpoint(latest_checkpoint_version)
                self.eval_policy_load_fn = lambda: self.eval_policy.unbundle(
                        checkpoint_dir, latest_checkpoint_version, experiment_data)
            else:
                raise NotImplementedError
        else:
            self.eval_policy_load_fn = lambda: None

        self._replay_scheme = replay_scheme
        self._double_dqn = double_dqn
        self._debug = debug
        self._num_atoms = num_atoms
        self._policy = policy
        self._target_estimator = target_estimator
        self._delta = delta
        self._action_sampler = ParticlePolicy(delta)
        self._action_sampler.use_marginal = use_marginal
        self._beta_ph = tf.placeholder(tf.float32, (), name='beta_schedule')
        self.beta_fn = beta_fn
        self.bandwidth_annealing_fn = bandwidth_annealing_fn
        self.h_ph = tf.placeholder(tf.float32, (), name='h')
        self.bandwidth_selection_type = bandwidth_selection_type
        self.kappa = kappa

        print(f'=== use_marginal: {self._action_sampler.use_marginal} ===')

        if debug:
            self.statistics_collection = []

        from dopamine.discrete_domains.reward_logger_update import reward_logger
        self.reward_dim = reward_logger.reward_dim

        print('==== creating agent: mmdqn_nd ====')
        print(f'==== reward_dim: {self.reward_dim} ====')

        self.create_sample_dict()

        if self.constrain_env:
            with tf.device(tf_device):
                self.constrain_ph = tf.compat.v1.placeholder(tf.float32, [self.reward_dim],
                                                             name='constrain_ph')
                self.constrain_less_ph = tf.compat.v1.placeholder(tf.bool, [self.reward_dim],
                                                                  name='constrain_less_ph')

        super(MMDAgent, self).__init__(
            sess=sess,
            num_actions=num_actions,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size,
            network=network,
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_fn=epsilon_fn,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            tf_device=tf_device,
            use_staging=use_staging,
            optimizer=optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency,
            save_vector_reward=True
        )

    def create_sample_dict(self):
        self.sample_dict = {'states': [],
                            'actions': [],
                            'rewards': [],
                            'next_states': [],
                            'losses': []}
        self.reward_set = set()

    def try_append_sample_dict(self, states, actions, rewards, next_states, losses):
        for state, action, reward, next_state, loss in zip(states, actions, rewards, next_states, losses):
            int_reward = tuple(np.ceil(reward * 10).astype(np.int))
            if int_reward not in self.reward_set:
                self.sample_dict['states'].append(state)
                self.sample_dict['actions'].append(action)
                self.sample_dict['rewards'].append(reward)
                self.sample_dict['next_states'].append(next_state)
                self.sample_dict['losses'].append(loss)
                self.reward_set.add(int_reward)

    def joint_value_distribution(self, state, action):
        # compute the prediction of joint value distribution on the fixed starting state.
        if self.maze_env:
            state_agent = np.expand_dims(state, axis=[0, -1])
        else:
            state_agent = np.zeros_like(self.state)
            state = np.reshape(state, self.observation_shape)
            state_agent[0, ..., -1] = state
        particles = self._sess.run(self._net_outputs.particles[0],
                                   {self.state_ph: state_agent})[action]
        return particles  # (sample_num, reward_dim)

    def joint_value_distribution_stacked(self, state, action):
        # compute the prediction of joint value distribution on the state with stacked K frames.
        particles = self._sess.run(self._net_outputs.particles[0], {self.state_ph: state})[action]
        return particles

    def _create_network(self, name):
        return self.network(num_actions=self.num_actions, num_samples=self._num_atoms,
                            reward_dim=self.reward_dim, name=name)

    def _build_replay_buffer(self, use_staging):
        if self._replay_scheme not in ['uniform', 'prioritized']:
            raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
        if self.constrain_env:
            reward_shape = (self.reward_dim * 2,)
        else:
            reward_shape = (self.reward_dim,)
        return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype,
            reward_shape=reward_shape
        )

    def _store_transition(self,
                          last_observation,
                          action,
                          reward,
                          is_terminal,
                          priority=None):
        # print(f'reward in store transition: {reward}')
        if priority is None:
            if self._replay_scheme == 'uniform':
                priority = 1.
            else:
                priority = self._replay.memory.sum_tree.max_recorded_priority

        if not self.eval_mode:
            self._replay.add(last_observation, action, reward, is_terminal, priority)

    def _build_target_particles(self):
        batch_size = self._replay.batch_size
        rewards = self._replay.rewards[:, None, None, :]  # (bs,1,1,k)
        if self.constrain_env:
            remained_constraint = rewards[:, :, :, self.reward_dim:]  # (bs,1,1,k)
            rewards = rewards[:, :, :, :self.reward_dim]
            print('tensor remained_constraint:', remained_constraint)
        else:
            remained_constraint = None
        is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
        gamma_with_terminal = gamma_with_terminal[:, None, None, None]  # (bs,1,1,1)
        next_particles = self._replay_next_target_net_outputs.particles  # (bs,na,n,k)
        target_particles = rewards + gamma_with_terminal * next_particles  # (bs,na,n,k)
        print('particles: ', next_particles.shape, rewards.shape, gamma_with_terminal.shape, target_particles.shape)
        if self.evaluation_setting and self.eval_policy is not None:
            eval_policy_q_values = self.eval_policy.evaluate_states(self._replay.next_states)  # TODO: is it next_state?
            print(eval_policy_q_values.shape)
            eval_policy_probs = tf.one_hot(tf.argmax(eval_policy_q_values, axis=1), self.num_actions)
            eval_policy_probs = tf.ones_like(eval_policy_probs) / self.num_actions * self.eval_policy_epsilon + \
                                eval_policy_probs * (1 - self.eval_policy_epsilon)  # TODO:
        else:
            eval_policy_probs = None
        if self.constrain_env:
            estimator = 'constrain'

        else:
            estimator = self._target_estimator
        return self._action_sampler.compute_target(target_particles, estimator=estimator,
                                                   evaluation_setting=self.evaluation_setting,
                                                   eval_policy_probs=eval_policy_probs,
                                                   constraint=remained_constraint)

    def _build_networks(self):
        self.online_convnet = self._create_network(name='Online')
        self.target_convnet = self._create_network(name='Target')
        self._net_outputs = self.online_convnet(self.state_ph)

        # action_prob = tf.cast(tf.equal(tf.reduce_max(self._net_outputs.q_values, axis=1)[:,None], self._net_outputs.q_values), tf.float32)
        # action_prob = action_prob / tf.reduce_sum(action_prob, axis=-1)
        # self._q_argmax = tfp.distributions.Categorical(probs=action_prob).sample(1)[:,0][0]
        if self.constrain_env:
            self._q_argmax = self._action_sampler.draw_action(self._net_outputs.particles, 'constrain',
                                                              constrain_ph=self.constrain_ph,
                                                              constrain_less_ph=self.constrain_less_ph)[0]
        else:
            self._q_argmax = self._action_sampler.draw_action(self._net_outputs.particles, 'mean')[0]
        self._q_argmax_explore = \
        self._action_sampler.draw_action(self._net_outputs.particles, self._policy, beta=self._beta_ph)[0]
        self._replay_net_outputs = self.online_convnet(self._replay.states)
        self._replay_next_target_net_outputs = self.target_convnet(self._replay.next_states)

    def _build_train_op(self):  # *Required.
        target_particles = tf.stop_gradient(self._build_target_particles())  # (bs,n,k)
        indices = tf.range(tf.shape(self._replay_net_outputs.particles)[0])[:, None]  # (bs,1)
        reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)  # (bs,2)
        chosen_action_particles = tf.gather_nd(self._replay_net_outputs.particles, reshaped_actions)  # (bs,n,k)

        # h = tf.gather_nd(self._replay_net_outputs.h, reshaped_actions)[:,None,None] + 1e-5 #(b,)

        if self.kappa == 0:
            d1 = tf.square(chosen_action_particles[:, :, None, :] - chosen_action_particles[:, None, :, :])  # (b,n,n,k)
            d2 = tf.square(chosen_action_particles[:, :, None, :] - target_particles[:, None, :, :])  # (b,n,n,k)
            d3 = tf.square(target_particles[:, :, None, :] - target_particles[:, None, :, :])  # (b,n,n,k)
        else:
            d1 = hubber_loss(chosen_action_particles[:, :, None, :] - chosen_action_particles[:, None, :, :],
                             kappa=self.kappa)  # (b,n,n)
            d2 = hubber_loss(chosen_action_particles[:, :, None, :] - target_particles[:, None, :, :],
                             kappa=self.kappa)  # (b,n,n)
            d3 = hubber_loss(target_particles[:, :, None, :] - target_particles[:, None, :, :], kappa=self.kappa)  # (b,n,n)

        tf.logging.info('Bandwith type: {}.'.format(self.bandwidth_selection_type))

        if self.bandwidth_selection_type == 'mixture':  # Mixute of kernels from MMD GAN
            batch_size = chosen_action_particles.shape.as_list()[0]

            # sigmas = [1e-3, 1e-2, 1e-1, 1, 2, 4, 5, 10, 20]
            # sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 5, 10, 15, 20, 25, 30, 100]
            # sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 5, 10, 15, 20, 25, 30, 50, 80, 100]
            # sigmas = [1.]
            if self.bandwidth_type == 'v0':
                sigmas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            elif self.bandwidth_type == 'v1':
                sigmas = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.64, 0.80, 0.96]
            elif self.bandwidth_type == 'v2':
                sigmas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.64, 0.80, 0.96,
                          2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
            elif self.bandwidth_type == 'v21':
                sigmas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.64, 0.80, 0.96]
            elif self.bandwidth_type == 'v22':  # experiment-619
                sigmas = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.64, 0.80, 0.96, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
            elif self.bandwidth_type == 'v3':  # experiment-619
                sigmas = [2 ** i for i in range(-8, 9)]
            elif self.bandwidth_type == 'v31':
                sigmas = [2 ** i for i in range(-9, 9)]
            elif self.bandwidth_type == 'v4':
                sigmas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
            elif self.bandwidth_type == 'v5':
                sigmas = [4 ** i for i in range(-8, 9)]
            else:
                raise NotImplementedError
            # sigmas = [0.1,0.3,1,3,10]
            # sigmas = [1,2,4,8,10]
            # sigmas = [1,2,3,4,5]

            if self.kscale_type == 'v0':
                kscales = [1.0 for b in sigmas]
            elif self.kscale_type == 'v11':
                kscales = [b ** 0.5 for b in sigmas]
            elif self.kscale_type == 'v12':
                kscales = [b ** 0.3 for b in sigmas]
            else:
                raise NotImplementedError

            print(f'squared bandwidth: {sigmas}')
            print(f'kernel scales: {kscales}')

            h = sigmas[-1]  # for debug only
            xixj = gaussian_rbf_kernel(d1, sigmas, kscales)
            xixj = tf.reduce_mean(tf.reduce_mean(xixj, axis=-1), axis=-1)  # (b.)

            xiyj = gaussian_rbf_kernel(d2, sigmas, kscales)
            xiyj = tf.reduce_mean(tf.reduce_mean(xiyj, axis=-1), axis=-1)  # (b,)

            yiyj = gaussian_rbf_kernel(d3, sigmas, kscales)
            yiyj = tf.reduce_mean(tf.reduce_mean(yiyj, axis=-1), axis=-1)
            mmd_squared = xixj + yiyj - 2 * xiyj

            mmd_squared = tf.compat.v1.where(mmd_squared > 0, mmd_squared, tf.constant([0.] * batch_size), name='value')

            if self.use_priority:
                probs = self._replay.transition['sampling_probabilities']
                loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
                loss_weights /= tf.reduce_max(loss_weights)

                update_priorities_op = self._replay.tf_set_priority(
                    self._replay.indices, tf.sqrt(mmd_squared + 1e-10))
                mmd_squared = loss_weights * mmd_squared
            else:
                update_priorities_op = tf.no_op()

            loss = tf.reduce_mean(mmd_squared)

            train_op = self.optimizer.minimize(loss)
        else:
            if self.bandwidth_selection_type == 'med':  # Use median trick
                batch_size = chosen_action_particles.shape.as_list()[0]
                # aggregated_d = d1
                aggregated_d = tf.concat([tf.reshape(d1[:, :, 0], [batch_size, -1]), tf.reshape(d2, [batch_size, -1]),
                                          tf.reshape(d3[:, :, 0], [batch_size, -1])], axis=1)
                # aggregated_d = tf.concat(  [ tf.reshape(d1[:,:,0], [batch_size,-1]), tf.reshape(d2, [batch_size, -1])], axis=1)

                med_sq = tfp.stats.percentile(tf.reshape(aggregated_d, [batch_size, -1]), q=50., axis=1,
                                              interpolation='midpoint')  # (bs,)
                # # h = med_sq / tf.log(2*self._num_atoms + 1.)
                h = 2 * med_sq  # * 0.002
                h = tf.stop_gradient(h[:, None, None]) + 1e-5
            elif self.bandwidth_selection_type == 'annealing':
                h = self.h_ph  # 20.
            elif self.bandwidth_selection_type == 'const':
                h = 1.
            else:
                raise ValueError('Unrecognized bandwidth type: {}.'.format(self.bandwidth_selection_type))

            xixj = tf.reduce_mean(tf.reduce_mean(tf.exp(-d1 / h), axis=-1),
                                  axis=-1)  # / (self._num_atoms * (self._num_atoms-0)) # (b.)
            xiyj = tf.reduce_mean(tf.reduce_mean(tf.exp(-d2 / h), axis=-1), axis=-1)  # (b,)
            yiyj = tf.reduce_mean(tf.reduce_mean(tf.exp(-d3 / h), axis=-1), axis=-1)
            mmd = xixj + yiyj - 2 * xiyj
            loss = tf.reduce_mean(mmd)

            train_op = self.optimizer.minimize(loss)

        ## Monitor
        debug_particles = self._replay_net_outputs.particles  # (bs,na,n)
        p_std = tf.reduce_mean(tf.math.reduce_std(debug_particles, axis=-1))
        p_mean = tf.reduce_mean(tf.reduce_mean(debug_particles, axis=-1))
        p_min = tf.reduce_mean(tf.reduce_min(debug_particles, axis=-1))
        p_max = tf.reduce_mean(tf.reduce_max(debug_particles, axis=-1))
        debug_var = [p_min, p_max, p_mean, p_std, loss, tf.constant(h)]

        self.loss = mmd_squared

        with tf.control_dependencies([update_priorities_op]):
            if self.summary_writer is not None:
                with tf.variable_scope('Losses'):
                    tf.summary.scalar('1d_MMD_Loss', tf.reduce_mean(loss))

            return loss, debug_var, train_op  # train_op

    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
        (1) A minimum number of frames have been added to the replay buffer.
        (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.memory.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                # for _ in range(10):
                #     self._sess.run(self._train_op[-1])
                # loss,debug_v, _ = self._sess.run([self._train_op[0], self._train_op[1], self._train_op[2]])
                h = self.bandwidth_annealing_fn(self.training_steps)
                loss, debug_v, _ = self._sess.run(self._train_op, feed_dict={self.h_ph: h})

                if self._debug:
                    self.statistics_collection.append(debug_v)
                if (self.summary_writer is not None and
                        self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary = self._sess.run(self._merged_summaries)
                    self.summary_writer.add_summary(summary, self.training_steps)

            if self.training_steps % self.target_update_period == 0:
                self._sess.run(self._sync_qt_ops)

        self.distribution_log()
        self.training_steps += 1

    def distribution_log(self):
        if self._replay.memory.add_count > self.min_replay_history:
            collect_steps = 51000 # 249999
            if collect_steps - 1000 <= self.training_steps % 250000 <= collect_steps:
                states, actions, rewards, next_states, losses = self._sess.run([
                    self._replay.states, self._replay.actions,
                    self._replay.rewards, self._replay.next_states, self.loss
                ])
                if self.constrain_env:
                    rewards = rewards[..., :self.reward_dim]
                self.try_append_sample_dict(states, actions, rewards, next_states, losses)
            if self.training_steps % 250000 == collect_steps:
                states, actions, rewards, next_states, losses = \
                    self.sample_dict['states'], self.sample_dict['actions'], \
                    self.sample_dict['rewards'], self.sample_dict['next_states'], self.sample_dict['losses']
                states, actions, rewards, next_states, losses = map(lambda arr: np.stack(arr),
                                                            [states, actions, rewards, next_states, losses])
                actions_one_hot = tf.one_hot(actions, self.num_actions)

                num_samples = 1000
                samples = self._sess.run(
                    self.online_convnet.generate_samples(states, actions_one_hot, num_samples=num_samples)
                )  # [batch_size, num_samples, reward_dim]

                netoutputs = self.target_convnet(next_states, num_samples=num_samples)
                q_samples, q_values = netoutputs.particles, netoutputs.q_values

                # q_samples: [batch_size, num_actions, num_samples, reward_dim]
                # q_values: [batch_size, num_actions]

                actions_max = tf.argmax(q_values, axis=1)
                actions_max_one_hot = tf.one_hot(actions_max, self.num_actions)
                next_samples = tf.reduce_sum(
                    tf.expand_dims(tf.expand_dims(actions_max_one_hot, axis=-1), axis=-1) * q_samples,
                    axis=1
                )  # [batch_size, num_samples, reward_dim]

                next_samples = self._sess.run(next_samples)

                if states.ndim == 5:
                    states = states[..., 0]
                    next_states = next_states[..., 0]

                distribution_logger.log(samples, next_samples, states, actions, rewards, next_states, losses,
                                        self.cumulative_gamma)

                self.create_sample_dict()

    def begin_episode(self, observation, **kwargs):
        self.is_begin_episode = True
        self.initial_value = None
        self.action = super(MMDAgent, self).begin_episode(observation, **kwargs)
        self.is_begin_episode = False
        if self.eval_policy is not None:
            self.action = self.eval_policy.begin_episode(observation)
        return self.action

    def end_episode(self, reward, **kwargs):
        if self.constrain_env:
            remained_constraint = kwargs['remained_constraint']
            reward = np.concatenate([reward, remained_constraint], axis=0)
        super(MMDAgent, self).end_episode(reward)

    def step(self, reward, observation, **kwargs):
        if self.constrain_env:
            remained_constraint = kwargs['remained_constraint']
            reward = np.concatenate([reward, remained_constraint], axis=0)
        self.action = super(MMDAgent, self).step(reward, observation, **kwargs)
        if self.evaluation_setting:
            if self.eval_policy is None:
                self.action = random.choice(range(self.num_actions))
            else:
                self.action = self.eval_policy.step(reward, observation)
        return self.action

    def _select_action(self, **kwargs):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
            int, the selected action.
        """
        if self._policy == 'eps_greedy':
            if self.eval_mode:
                epsilon = self.epsilon_eval
            else:
                epsilon = self.epsilon_fn(
                    self.epsilon_decay_period,
                    self.training_steps,
                    self.min_replay_history,
                    self.epsilon_train)
            if random.random() <= epsilon:
                # Choose a random action with probability epsilon.
                return random.randint(0, self.num_actions - 1)
            else:
                # Choose the action with highest Q-value at the current state.
                # print(f'is_less here: {kwargs["is_less_constraint"]}')
                if self.constrain_env:
                    feed_dict = {
                        self.state_ph: self.state,
                        self.constrain_ph: kwargs['remained_constraint'],
                        self.constrain_less_ph: kwargs['is_less_constraint']
                    }
                else:
                    feed_dict = {self.state_ph: self.state}
                return self._sess.run(self._q_argmax, feed_dict)
        else:
            if self.eval_mode:
                epsilon = self.epsilon_eval
                if random.random() <= epsilon:
                    return random.randint(0, self.num_actions - 1)
                else:
                    if self._policy in ['ucb', 'ps']:
                        return self._sess.run(self._q_argmax, {self.state_ph: self.state})
                    else:  # 'ps2', 'ps3'
                        return self._sess.run(self._q_argmax, {self.state_ph: self.state})
            else:
                beta = self.beta_fn(self.training_steps)
                return self._sess.run(self._q_argmax_explore, {self.state_ph: self.state, self._beta_ph: beta})


class ParticlePolicy(object):
    def __init__(self, delta=0.1, quantile_index=None):
        """
        Args:
            target_type: str, [mode/separate]
        """
        self.delta = delta
        self.beta = delta  # norm.ppf(1 - delta, loc=0, scale=1) Too big beta might explode the learning
        self.quantile_index = quantile_index

    @staticmethod
    def compute_thompson_matrix(particles):
        """Compute Thompson probability matrix.
        Args:
            particles: (bs,na,n)
        Returns:
            logits: (bs,na)
        """
        shape = particles.shape.as_list()
        bs = shape[0]
        na = shape[1]
        n = shape[2]
        indices = tf.range(n)
        logits = []
        for i in range(na):
            q1 = particles[:, i, :]
            i_index = tf.constant(np.array([j for j in range(na) if j != i]))
            q2 = tf.gather(particles, i_index, axis=1)
            s = tf.cast(tf.greater_equal(q1[:, None, None, :], q2[:, :, :, None]), dtype=tf.float32)
            logits.append(tf.reduce_sum(tf.math.reduce_prod(tf.reduce_sum(s, axis=2), axis=1), axis=1))
        logits = tf.stack(logits, axis=1)
        return logits

    @staticmethod
    def sample_from_action_probability(action_values):
        """
        Args:
            action_values: (bs,na)

        Returns:
            selected_action: (bs,), one of the actions with maximum value.
        """
        action_prob = tf.cast(tf.equal(tf.reduce_max(action_values, axis=1)[:, None], action_values), tf.float32)
        # selected_action = tf.random.categorical(logits=action_prob, num_samples=1)[:,0] # FLAG: logits=[0,1,1] -> 0.16 0.42 0.42
        action_prob = action_prob / tf.reduce_sum(action_prob, axis=-1)
        selected_action = tfp.distributions.Categorical(probs=action_prob).sample(1)[:, 0]
        return selected_action  # tf.squeeze(selected_action)

    def draw_action(self, particles, policy='mean', head_index=0, random_weights=np.array([1]), beta=None,
                    constrain_ph=None, constrain_less_ph=None):
        """Compute selected action based on the approximate posterior particles.
        Args:
            particles: (bs,na,n)
            policy: str, [eps_greedy/mean/ucb/ps/boot/ensemble]. [mean/optimistic/posterior] for target estimator.
            head_index: int (for boot policy)
            random_weights:

        Returns:
            selected_action: (bs,)
        """
        if policy in ['eps_greedy', 'mean']:
            q_values = tf.reduce_mean(particles, axis=[2, 3])

            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)
        elif policy in ['constrain']:
            # particles: [B, A, N, K]
            print('!!!!!!!!!!! in constrain taking action')
            greater = tf.greater_equal(particles, constrain_ph[None, None, None, :])
            satisfy_constraint = tf.math.logical_xor(greater, constrain_less_ph[None, None, None, :])
            if self.use_marginal:
                #  prob_satisfy_constraint: [B, A, K]
                prob_each_satisfy_constraint = tf.reduce_mean(tf.cast(satisfy_constraint, tf.float32), axis=2)
                log_prob_satisfy_constraint = tf.math.log(prob_each_satisfy_constraint + 1e-7)
                log_prob_joint_satisfy_constraint = tf.reduce_sum(log_prob_satisfy_constraint, axis=-1)
                prob_satisfy_constraint = log_prob_joint_satisfy_constraint
            else:
                all_satisfy_constraint = tf.math.reduce_all(satisfy_constraint, axis=-1)
                all_satisfy_constraint = tf.cast(all_satisfy_constraint, tf.float32)
                prob_satisfy_constraint = tf.reduce_mean(all_satisfy_constraint, axis=-1)  # [B, A]
                self.particles = particles
                self.greater = greater
                self.satisfy_constraint = satisfy_constraint
                self.all_satisfy_constraint = all_satisfy_constraint
            self.prob_satisfy_constraint = prob_satisfy_constraint
            return self.sample_from_action_probability(prob_satisfy_constraint)

        elif policy in ['ucb', 'optimistic']:
            q_mean_values = tf.reduce_mean(particles, axis=2)  # (bs,na)
            if self.quantile_index is None:
                q_std_values = tf.math.reduce_std(particles, axis=2)  # (bs,na)
                if beta is None:
                    beta = self.beta
                q_values = q_mean_values + beta * q_std_values
            else:
                q_values = q_mean_values + particles[:, :, self.quantile_index]
            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)
        elif policy in ['ucb_max', 'optimistic_max']:
            q_mean_values = tf.reduce_mean(particles, axis=2)  # (bs,na)
            q_values = q_mean_values + tf.reduce_max(particles, axis=-1)
            return self.sample_from_action_probability(q_values)

        elif policy in ['ps', 'posterior']:
            # A head is sampled at each time step, as opposed to bootrapped policy where a head is sampled at each episode.
            p_shape = particles.shape.as_list()
            logits = tf.ones([p_shape[0] * p_shape[1], p_shape[2]], dtype=tf.float32)
            indices = tf.reshape(tf.random.categorical(logits, num_samples=1), [p_shape[0], p_shape[1]])  # (bs,na)
            mask = tf.one_hot(indices, depth=p_shape[2])  # (bs,na,n) where the last dim is one-hot
            q_values = tf.reduce_sum(tf.multiply(particles, mask), axis=2)
            return self.sample_from_action_probability(q_values)
        elif policy in ['ps2', 'posterior2']:
            # Equation (3) in B. O'Donoghue et al. "The Uncertainty Bellman Equation and Exploration".
            p_shape = particles.shape.as_list()
            q_mean_values = tf.reduce_mean(particles, axis=2)  # (bs,na)
            q_std_values = tf.math.reduce_std(particles, axis=2)  # (bs,na)
            beta = tf.random.normal((p_shape[0], p_shape[1]))  # (bs,na)
            q_values = q_mean_values + beta * q_std_values
            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)
        elif policy in ['ps3', 'posterior3']:
            # Random weight of the heads into a randomized Q-function
            p_shape = particles.shape.as_list()
            random_ensemble_weights = tf.random.normal((p_shape[-1],))  # (bs,na)
            random_ensemble_weights = random_ensemble_weights / tf.reduce_sum(random_ensemble_weights)
            q_values = tf.reduce_mean(tf.multiply(particles, random_ensemble_weights[None, None, :]), axis=-1)
            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)
        elif policy == 'boot':
            # Select a head uniformaly at random at the start of the episode and follow this choice for an entire episode.
            # Q: How about in evaluation?
            q_values = particles[:, :, head_index]  # (bs,na)
            # print('DEBUGGGG')
            # print(q_values)
            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)

        elif policy == 'rem':
            # Inspired by https://arxiv.org/abs/1907.04543v3, randomly combine the heads into a randomized head.
            q_values = tf.reduce_sum(tf.multiply(particles, random_weights[None, None, :]), axis=-1)
            return self.sample_from_action_probability(q_values)

        elif policy == 'ensemble':
            # Choose action based on the majority vote across heads
            # Q: episode-based or step-based? Seems step-based is more natural.
            argmax_ensemble = tf.math.argmax(particles, axis=1)  # (bs,n)
            assert argmax_ensemble.shape.as_list()[0] == 1
            argmax_ensemble = tf.squeeze(argmax_ensemble)  # (n,)
            with tf.device('/cpu:0'):  # tf.unique_with_counts is not supported in GPU.
                y, idx, count = tf.unique_with_counts(argmax_ensemble)
                max_count_idx = tf.math.argmax(count)
                # print('DEBUGGG')
                # print(y)
                # print(max_count_idx)
                # print(argmax_ensemble)
                return tf.gather(y, max_count_idx)[None]
        else:
            raise ValueError('Unrecognized policy: {}'.format(policy))

    def compute_target(self, targets, estimator='mean', random_weights=np.array([1]), evaluation_setting=False,
                       eval_policy_probs=None, constraint=None):
        """
        Args:
            targets: (bs,na,n)
            estimator: str, [mean/optimistic/posterior/head_wise]

        Returns:
            action_targets: (bs,n)
        """
        if estimator == 'mean':
            q_values = tf.reduce_mean(targets, axis=[2,3])  # (bs,na)
            if evaluation_setting:
                if eval_policy_probs is None:
                    actions = tf.random.uniform(shape=[q_values.shape[0]], maxval=q_values.shape[1], dtype=tf.int32)
                else:
                    log_prob = tf.math.log(eval_policy_probs)
                    actions = tf.random.categorical(log_prob, 1)[:, 0]
                    print('actions: ', actions)
                action_prob = tf.one_hot(actions, q_values.shape[1])
            else:
                action_prob = tf.cast(tf.equal(tf.reduce_max(q_values, axis=1)[:, None], q_values), tf.float32)  # (bs,na)
                action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:, :, None, None]), axis=1)  # (bs,n,k)
            print(action_prob.shape, targets.shape)
            print('############## returning target: ', action_targets.shape)
            return action_targets
        elif estimator == 'constrain':
            print('in constrain estimator')
            greater = tf.greater_equal(targets, constraint)  # (bs,na,N,K)
            is_less_constraint = tf.constant([False, False, False])
            satisfy_constraint = tf.math.logical_xor(greater, is_less_constraint[None, None, None, :])
            if self.use_marginal:
                #  prob_satisfy_constraint: [B, A, K]
                prob_each_satisfy_constraint = tf.reduce_mean(tf.cast(satisfy_constraint, tf.float32), axis=2)
                log_prob_satisfy_constraint = tf.math.log(prob_each_satisfy_constraint + 1e-7)
                log_prob_joint_satisfy_constraint = tf.reduce_sum(log_prob_satisfy_constraint, axis=-1)
                prob_satisfy_constraint = log_prob_joint_satisfy_constraint
                self.t_particles = targets
                self.t_constraint = constraint
                self.t_greater = greater
                self.t_satisfy_constraint = satisfy_constraint
                self.t_all_satisfy_constraint = prob_each_satisfy_constraint
                self.t_prob_satisfy_constraint = prob_satisfy_constraint
            else:
                all_satisfy_constraint = tf.math.reduce_all(satisfy_constraint, axis=-1)
                all_satisfy_constraint = tf.cast(all_satisfy_constraint, tf.float32)
                prob_satisfy_constraint = tf.reduce_mean(all_satisfy_constraint, axis=-1)  # [B, A]
                self.t_particles = targets
                self.t_constraint = constraint
                self.t_greater = greater
                self.t_satisfy_constraint = satisfy_constraint
                self.t_all_satisfy_constraint = all_satisfy_constraint
                self.t_prob_satisfy_constraint = prob_satisfy_constraint
            action_prob = tf.cast(tf.equal(tf.reduce_max(prob_satisfy_constraint, axis=1)[:, None],
                                           prob_satisfy_constraint), tf.float32)  # (bs,na)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:, :, None, None]), axis=1)  # (bs,n,k)

            return action_targets

        elif estimator == 'optimistic':
            q_mean_values = tf.reduce_mean(targets, axis=2)  # (bs,na)
            if self.quantile_index is None:
                q_std_values = tf.math.reduce_std(targets, axis=2)  # (bs,na)
                q_values = q_mean_values + self.beta * q_std_values
            else:
                q_values = q_mean_values + targets[:, :, self.quantile_index]

            action_prob = tf.cast(tf.equal(tf.reduce_max(q_values, axis=1)[:, None], q_values), tf.float32)  # (bs,na)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:, :, None]), axis=1)
            return action_targets
        elif estimator == 'optimistic_max':
            q_mean_values = tf.reduce_mean(targets, axis=2)  # (bs,na)
            q_values = q_mean_values + tf.reduce_max(targets, axis=-1)

            action_prob = tf.cast(tf.equal(tf.reduce_max(q_values, axis=1)[:, None], q_values), tf.float32)  # (bs,na)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:, :, None]), axis=1)
            return action_targets
        elif estimator == 'posterior':
            action_prob = self.compute_thompson_matrix(targets)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:, :, None]), axis=1)
            return action_targets
        elif estimator == 'head_wise':
            action_targets = tf.reduce_max(targets, axis=1)
            return action_targets
        elif estimator == 'posterior3':
            p_shape = targets.shape.as_list()
            random_ensemble_weights = tf.random.normal((p_shape[-1],))  # (bs,na)
            random_ensemble_weights = random_ensemble_weights / tf.reduce_sum(random_ensemble_weights)
            q_values = tf.reduce_mean(tf.multiply(targets, random_ensemble_weights[None, None, :]), axis=-1)
            action_prob = tf.cast(tf.equal(tf.reduce_max(q_values, axis=1)[:, None], q_values), tf.float32)  # (bs,na)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:, :, None]), axis=1)
            return action_targets
        else:
            raise ValueError('Unrecognized estimator: {}.'.format(estimator))
