import asyncio
import concurrent.futures

import tensorflow as tf
import numpy as np
from tf_agents.trajectories import from_transition
from reference.specs import time_step_spec





class MappedDriver:
	def __init__(self, environment, policy_list, policy_mapping, observers, max_steps=np.inf, max_episodes=np.inf):
		self.environment = environment
		self.policy_list = policy_list
		self.observers = observers or []
		self.policy_mapping = policy_mapping
		self.action_mapping = self.get_action_mapping(policy_mapping)
		self.max_steps = max_steps
		self.max_episodes = max_episodes
		self.action_shape = None

	def get_action_mapping(self, policy_mapping):
		return tf.argsort(tf.reshape(policy_mapping, (-1,)))

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

	def get_action(self, mapped_time_step):
		# with concurrent.futures.ThreadPoolExecutor() as executor:
		# 	futures = [
		# 		executor.submit(policy.action, time_step)
		# 		for time_step, policy in zip(mapped_time_step, self.policy_list)
		# 	]
		# 	results = [future.result() for future in futures]
		# 	print(results)
		# 	return results

		# # action_list = [None] * len(self.policy_list)
		# with concurrent.futures.ThreadPoolExecutor() as executor:
		# 	futures = []
		# 	for index, (policy, time_step) in enumerate(zip(self.policy_list, mapped_time_step)):
		# 		futures.append(executor.submit(policy.action, time_step=time_step))
		# 	# for i, future in enumerate(futures):
		# 	# 	# result = future.result()
		# 	# 	# print(result)
		# 	# 	action_list[i] = future.result()
		#
		# print(action_list)
		action_list = []
		for index, (policy, time_step) in enumerate(zip(self.policy_list, mapped_time_step)):
			action_list.append(policy.action(time_step=time_step))

		return action_list

	def map_action(self, action_list):
		action_list = np.array([a.action for a in action_list])
		action_list = tf.reshape(action_list, (-1,))
		action_list = tf.gather(action_list, self.action_mapping)
		action_list = tf.reshape(action_list, self.action_shape)
		return action_list

	def get_time_step(self, action):
		env_action = self.map_action(action)
		env_time_step = self.environment.step(env_action)
		next_time_step = self.map_time_step(env_time_step)
		return next_time_step

	def run(self, time_step=None):
		num_steps = 0
		num_episodes = 0

		if not time_step:
			time_step = self.environment.current_time_step()
			if self.action_shape is None:
				self.action_shape = time_step[0].shape[:-1]
			time_step = self.map_time_step(time_step)

		while num_steps < self.max_steps and num_episodes < self.max_episodes:
			action = self.get_action(time_step)
			next_time_step = self.get_time_step(action)

			traj_list = [from_transition(*info) for info in zip(time_step, action, next_time_step)]

			for observer in self.observers:
				observer(traj_list)

			for traj in traj_list:
				num_episodes += np.sum(traj.is_last())
				num_steps += np.sum(~traj.is_boundary())

			time_step = next_time_step
