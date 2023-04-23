import tensorflow as tf
import numpy as np

from tf_agents.trajectories import TimeStep
from reference.specs import time_step_spec


class MappedDriver:
	def __init__(self, environment, policy_list, policy_mapping, observers, max_steps=np.inf, max_episodes=np.inf):
		self.environment = environment
		self.policy_list = policy_list
		self.observers = observers or []
		self.policy_mapping = policy_mapping
		self.max_steps = max_steps
		self.max_episodes = max_episodes

	def map_time_step(self, time_step):
		def map_to_splits(a):
			a = tf.reshape(a, (-1, a.shape[-1]))
			a = tf.squeeze(a)
			a = tf.gather(a, self.policy_mapping)
			return a

		splits = tf.nest.map_structure(map_to_splits, time_step)
		splits = tf.nest.flatten(splits)
		splits = zip(*splits)
		splits = [tf.nest.pack_sequence_as(time_step_spec, split) for split in splits]
		return splits

	def map_action(self, mapped_time_step):
		action_list = []
		for policy, time_step in zip(self.policy_list, mapped_time_step):
			action_list.append(policy.action(time_step).action)
		return tf.convert_to_tensor(action_list, dtype=tf.int32)

	def run(self, time_step=None):
		num_steps = 0
		num_episodes = 0

		if not time_step:
			time_step = self.environment.current_time_step()
			time_step = self.map_time_step(time_step)

		while num_steps < self.max_steps and num_episodes < self.max_episodes:
			action = self.map_action(time_step)
			next_time_step = self.map_time_step(self.environment.step(action))

			traj_list = [TimeStep(*info) for info in zip(time_step, action, next_time_step)]

			for observer in self.observers:
				observer(traj_list)

			for traj in traj_list:
				num_episodes += np.sum(traj.is_last())
				num_steps += np.sum(~traj.is_boundary())
