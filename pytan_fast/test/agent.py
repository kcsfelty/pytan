import os
from abc import ABC
from typing import Callable
import numpy as np
import tensorflow as tf
from tf_agents.agents import DqnAgent, CategoricalDqnAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks.categorical_q_network import CategoricalQNetwork
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs import TensorSpec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import common

from pytan_fast.test import hyperparameters
from pytan_fast.test.hyperparameters import gamma, learning_rate, replay_buffer_size, training_num_steps, \
	training_batch_size

summaries_flush_secs = 10


class PytanAgent(CategoricalDqnAgent, ABC):

	def __init__(self,
				 env: TFPyEnvironment,
				 observation_spec: TensorSpec = None,
				 action_spec: TensorSpec = None,
				 reward_fn: Callable = lambda time_step: time_step.reward,
				 name: str = 'IMAgent',
				 player_index=-1,
				 num_atoms=51,
				 q_network=None,
				 train_metrics=None,
				 eval_metrics=None,
				 checkpoint_dir="",
				 global_step=None,
				 train_dir="",
				 eval_dir="",
				 # training params
				 replay_buffer_max_length: int = 1000,
				 learning_rate: float = 1e-5,
				 training_batch_size: int = 8,
				 training_parallel_calls: int = 3,
				 training_prefetch_buffer_size: int = 3,
				 training_num_steps: int = 2,
				 **dqn_kwargs):

		self._env = env
		self._reward_fn = reward_fn
		self._name = name
		self._player_index = player_index
		self._observation_spec = observation_spec or self._env.observation_spec()
		self._action_spec = action_spec or self._env.action_spec()
		# self._action_fn = self._get_action_fn()
		self._splitter_fn = self._get_splitter_fn()
		self._num_atoms = num_atoms
		self._train_metrics = train_metrics
		self._eval_metrics = eval_metrics
		self._global_step = global_step

		q_network = q_network or self._build_q_net()
		self._train_summary_writer = tf.compat.v2.summary.create_file_writer(
			train_dir, flush_millis=summaries_flush_secs * 1000)

		self._eval_summary_writer = tf.compat.v2.summary.create_file_writer(
			eval_dir, flush_millis=summaries_flush_secs * 1000)
		env_ts_spec = self._env.time_step_spec()
		time_step_spec = TimeStep(
			step_type=env_ts_spec.step_type,
			reward=self._env.reward_spec(),
			discount=env_ts_spec.discount,
			observation=self._env.observation_spec()
		)

		optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		super().__init__(time_step_spec,
						 self._action_spec,
						 q_network,
						 optimizer,
						 name=name,
						 observation_and_action_constraint_splitter=self._splitter_fn,
						 **dqn_kwargs)

		self._policy_state = self.policy.get_initial_state(
			batch_size=self._env.batch_size)
		self._rewards = []

		self._replay_buffer = TFUniformReplayBuffer(
			data_spec=self.collect_data_spec,
			batch_size=self._env.batch_size,
			max_length=replay_buffer_max_length)

		self._training_batch_size = training_batch_size
		self._training_parallel_calls = training_parallel_calls
		self._training_prefetch_buffer_size = training_prefetch_buffer_size
		self._training_num_steps = training_num_steps
		self.train = common.function(self.train)
		self._iterator = None
		self._train_checkpointer = common.Checkpointer(
			ckpt_dir=checkpoint_dir,
			agent=self,
			global_step=global_step,
			metrics=metric_utils.MetricsGroup(self._train_metrics, 'train_metrics'),
			max_to_keep=1
		)
		self._policy_checkpointer = common.Checkpointer(
			ckpt_dir=os.path.join(checkpoint_dir, 'policy'),
			policy=self.policy,
			global_step=global_step,
			max_to_keep=1
		)
		self._rb_checkpointer = common.Checkpointer(
			ckpt_dir=os.path.join(checkpoint_dir, 'replay_buffer'),
			max_to_keep=1,
			replay_buffer=self._replay_buffer,
		)

		print("Loading checkpoint for agent{}".format(player_index))
		self._train_checkpointer.initialize_or_restore()
		print("\tLoaded train checkpoint")
		self._rb_checkpointer.initialize_or_restore()
		print("\tLoaded replay buffer checkpoint")

	# def _get_action_fn(self):
	#
	# 	def _action_fn(action):
	# 		return {'action': action, 'player_index': self._player_index}
	#
	# 	return _action_fn

	def _get_splitter_fn(self):
		def splitter_fn(observation):
			obs = observation[self._player_index]['observation']
			mask = observation[self._player_index]['action_mask']
			return obs, mask

		return splitter_fn

	def _build_q_net(self):
		q_net = CategoricalQNetwork(
			self._observation_spec[self._player_index]["observation"],
			self._action_spec,
			num_atoms=self._num_atoms,
			fc_layer_params=hyperparameters.fc_layer_params,
		)
		q_net.create_variables()
		# q_net.summary()
		return q_net

	def reset(self):
		self._policy_state = self.policy.get_initial_state(
			batch_size=self._env.batch_size)
		self._rewards = []

	def episode_return(self) -> float:
		return np.sum(self._rewards)

	# def _augment_time_step(self, time_step: TimeStep) -> TimeStep:
	#
	# 	reward = self._reward_fn(time_step)
	# 	reward = tf.convert_to_tensor(reward, dtype=tf.float32)
	# 	if reward.shape != time_step.reward.shape:
	# 		reward = tf.reshape(reward, time_step.reward.shape)
	#
	# 	return TimeStep(
	# 		step_type=time_step.step_type,
	# 		reward=time_step.reward,
	# 		discount=time_step.discount,
	# 		observation=time_step.observation)
	# def _current_time_step(self) -> TimeStep:
	# 	time_step = self._env.current_time_step()
	# 	time_step = self._augment_time_step(time_step)
	# 	return time_step
	#
	# def _step_environment(self, action) -> TimeStep:
	# 	# action = self._action_fn(action)
	# 	# action = {'action': action, 'player_index': self._player_index}
	# 	time_step = self._env.step({'action': action, 'player_index': self._player_index})
	# 	time_step = self._augment_time_step(time_step)
	# 	return time_step

	def act(self, collect=False) -> Trajectory:
		# time_step = self._current_time_step()
		time_step = self._env.current_time_step()

		if collect:
			policy_step = self.collect_policy.action(
				time_step, policy_state=self._policy_state)
		else:
			policy_step = self.policy.action(
				time_step, policy_state=self._policy_state)

		self._policy_state = policy_step.state
		# next_time_step = self._step_environment(policy_step.action)
		next_time_step = self._env.step({'action': policy_step.action, 'player_index': self._player_index})
		traj = trajectory.from_transition(time_step, policy_step, next_time_step)

		self._rewards.append(next_time_step.reward)

		if collect:
			self._replay_buffer.add_batch(traj)

		return traj

	# .filter(_filter_invalid_transition)\
	def build_replay_iterator(self):
		dataset = self._replay_buffer.as_dataset(
			sample_batch_size=hyperparameters.training_batch_size,
			num_steps=2) \
			.unbatch() \
			.batch(hyperparameters.training_batch_size) \
			.prefetch(5)
		# Dataset generates trajectories with shape [Bx2x...]
		self._iterator = iter(dataset)

	def train_iteration(self) -> LossInfo:
		experience, info = self._replay_buffer.get_next(
			sample_batch_size=self._training_batch_size,
			num_steps=self._training_num_steps
		)
		with self._train_summary_writer.as_default():
			return self.train(experience)

	# experience, _ = next(self._iterator)
	# return self.train(experience)

	def write_episode_rewards(self, episode_index):
		with self._train_summary_writer.as_default():
			tf.summary.scalar(
				name="episode_rewards".format(self._player_index),
				data=self.episode_return(),
				step=episode_index
			)
			self._train_summary_writer.flush()

	def write_episode_metrics(self, traj, collect=False):
		if collect:
			if self._train_metrics:
				with self._train_summary_writer.as_default():
					tf.summary.scalar(name="agent{}/episode_rewards".format(self._player_index))
					for observer in self._train_metrics:
						observer(traj)
						observer.tf_summaries(train_step=self._global_step)
					self._train_summary_writer.flush()
		else:
			if self._eval_metrics:
				with self._eval_summary_writer.as_default():
					for observer in self._eval_metrics:
						observer(traj)
						observer.tf_summaries(train_step=self._global_step)
					self._eval_summary_writer.flush()

	def checkpoint_train(self):
		self._train_checkpointer.save(global_step=self._global_step.numpy())

	def checkpoint_policy(self):
		self._policy_checkpointer.save(global_step=self._global_step.numpy())

	def checkpoint_replay_buffer(self):
		self._rb_checkpointer.save(global_step=self._global_step.numpy())



checkpoint_base_dir = "C:\\Users\\manof\\PycharmProjects\\csci_pytan\\learning\\categorical"
tensorboard_base_dir = "C:\\Users\\manof\\PycharmProjects\\csci_pytan\\learning\\tensorboard"
train_dir = os.path.join(tensorboard_base_dir, 'train')
eval_dir = os.path.join(tensorboard_base_dir, 'eval')

def get_q_agents(player_index, get_epsilon):
	train_metrics = [
		tf_metrics.ChosenActionHistogram(
			name="ChosenActionHistogram_agent{}".format(player_index)),
		tf_metrics.AverageReturnMetric(
			prefix="Metrics".format(player_index))]

	eval_metrics = [
		tf_metrics.AverageEpisodeLengthMetric(),
		tf_metrics.AverageReturnMetric()]

	agent_train_dir = os.path.join(train_dir, "agent{}".format(player_index))
	agent_eval_dir = os.path.join(eval_dir, "agent{}".format(player_index))
	checkpoint_dir = os.path.join(checkpoint_base_dir, "agent{}".format(player_index))
	global_step = tf.compat.v1.train.get_or_create_global_step()
	agent = PytanAgent(
		train_env,
		name='Player{}'.format(player_index),
		action_spec=train_env.action_spec()['action'],
		observation_spec=base_env.observation_spec(),
		player_index=player_index,

		training_batch_size=training_batch_size,
		training_num_steps=training_num_steps,
		replay_buffer_max_length=replay_buffer_size,

		train_metrics=train_metrics,
		eval_metrics=eval_metrics,
		checkpoint_dir=checkpoint_dir,
		train_dir=agent_train_dir,
		eval_dir=agent_eval_dir,
		global_step=global_step,
		td_errors_loss_fn=common.element_wise_squared_loss,
		learning_rate=learning_rate,
		epsilon_greedy=get_epsilon,
		gamma=gamma,
		n_step_update=3,
		min_q_value=-1,
		max_q_value=1,
	)

	return agent
