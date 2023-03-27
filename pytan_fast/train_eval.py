import os
import time
from abc import ABC

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from pytan_fast.agent import FastAgent
from pytan_fast.game import PyTanFast
from pytan_fast.settings import player_count


def train_eval(
		env_specs,
		log_dir="./logs/",
		total_steps=100000000,

		# Training / Experience
		batch_size=500,
		replay_buffer_capacity=10000,

		# Hyperparameters
		fc_layer_params=(2**7, 2**5),
		learning_rate=0.001,
		n_step_update=5,

		# Intervals
		eval_interval=35000,
		train_interval=1,
		checkpoint_interval=10000,
	):
	global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
	global_step_checkpointer = common.Checkpointer(
		ckpt_dir=os.path.join("checkpoints", "global_step"),
		global_step=global_step,
		max_to_keep=1)
	global_step_checkpointer.initialize_or_restore()

	agent_list = [
		FastAgent(
			player_index=i,
			batch_size=batch_size,
			log_dir=log_dir,
			replay_buffer_capacity=replay_buffer_capacity,
			fc_layer_params=fc_layer_params,
			learning_rate=learning_rate,
			n_step_update=n_step_update,
			env_specs=env_specs,
			train_interval=train_interval,
			checkpoint_interval=checkpoint_interval,
			global_step=global_step,
			eval_interval=eval_interval,
			eps_decay_rate=1 - np.log(2) / 50000)
		for i in range(player_count)]

	# for external_data_term in ["0000"]:
	# 	replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
	# 		data_spec=agent_list[0].agent.collect_data_spec,
	# 		batch_size=1,
	# 		max_length=1000000)
	#
	# 	dataset = replay_buffer.as_dataset(
	# 		num_parallel_calls=3,
	# 		sample_batch_size=100,
	# 		num_steps=n_step_update + 1,
	# 	).prefetch(3)
	#
	# 	checkpointer = common.Checkpointer(
	# 		ckpt_dir=os.path.join("rb", external_data_term),
	# 		max_to_keep=1,
	# 		replay_buffer=replay_buffer)
	#
	# 	checkpointer.initialize_or_restore()
	# 	iterator = iter(dataset)
	#
	# 	print(external_data_term, "frames:", replay_buffer.num_frames())
	# 	for _ in range(1000):
	# 		data, _ = next(iterator)
	# 		for agent in agent_list:
	# 			with agent.writer.as_default():
	# 				agent.agent.train(data)

	def checkpoint():
		global_step_checkpointer.save(global_step=global_step.read_value())
		for checkpoint_agent in agent_list:
			checkpoint_agent.checkpoint()

	# class ParallelPyTanFast(PyTanFast, ABC):
	# 	def __init__(self):
	# 		super().__init__(agent_list, global_step, log_dir)
	# num_envs = 1
	# parallel_env = parallel_py_environment.ParallelPyEnvironment([ParallelPyTanFast] * int(num_envs))
	# lap_time = time.perf_counter()
	# log_interval = 50000
	env = PyTanFast(agent_list, global_step, log_dir)
	env.reset()

	try:
		while global_step.numpy() < total_steps:
			env.walk()

			# if global_step.numpy() % log_interval == 0:
			# 	step_rate = log_interval / (time.perf_counter() - lap_time)
			# 	lap_time = time.perf_counter()
			# 	print("Current step:", global_step.numpy().item(), "step rate:", int(step_rate))
			# 	global_step_checkpointer.save(global_step=global_step.read_value())

	except Exception as e:
		print("Exception caught, creating checkpoint...")
		print(e)
		checkpoint()
	except KeyboardInterrupt:
		print("KeyboardInterrupt caught, creating checkpoint...")
		checkpoint()


if __name__ == "__main__":
	_env = PyTanFast()
	train_env = tf_py_environment.TFPyEnvironment(_env)
	_env_specs = {
		"env_observation_spec": train_env.observation_spec()[0],
		"env_action_spec": train_env.action_spec()[0],
		"env_time_step_spec": train_env.time_step_spec(),
	}

	policy_half_life_steps = 10000
	decay_rate = np.log(2) / policy_half_life_steps

	train_eval(
		env_specs=_env_specs,
		learning_rate=decay_rate,
		eval_interval=policy_half_life_steps * 7,
		replay_buffer_capacity=policy_half_life_steps * 3,
	)
