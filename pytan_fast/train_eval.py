import os

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

from pytan_fast.agent import FastAgent
from pytan_fast.game import PyTanFast
from pytan_fast.random_agent import RandomAgent
from pytan_fast.settings import player_count


def train_eval(
		env_specs,
		log_dir="./logs/",
		total_steps=100000000,

		# Training / Experience
		replay_ratio=1,
		replay_buffer_capacity=10000,

		# Hyperparameters
		fc_layer_params=(2**7, 2**6),
		learning_rate=0.001,
		n_step_update=45,

		# Intervals
		eval_interval=35000,
		train_interval=50,
		checkpoint_interval=50000,
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
			batch_size=train_interval * replay_ratio,
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
			eps_decay_rate=1 - np.log(2) / 100000,
			min_train_frames=20000)
		for i in range(player_count)]

	def checkpoint():
		global_step_checkpointer.save(global_step=global_step.read_value())
		for checkpoint_agent in agent_list:
			checkpoint_agent.checkpoint()

	env = PyTanFast(agent_list, global_step, log_dir)
	env.reset()

	try:
		while global_step.numpy() < total_steps:
			env.walk()

			if global_step.numpy() % checkpoint_interval == 0:
				global_step_checkpointer.save(global_step=global_step.read_value())

	except Exception as e:
		print("Exception caught, creating checkpoint...")
		print(e)
		checkpoint()
	except KeyboardInterrupt:
		print("KeyboardInterrupt caught, creating checkpoint...")
		checkpoint()


def random_experience(env_specs, agent_list, log_dir):
	random_agent_list = [RandomAgent(
		env_specs=env_specs,
		player_index=agent.player_index,
		observers=[agent.replay_buffer.add_batch]
	) for agent in agent_list]
	global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
	env = PyTanFast(random_agent_list, global_step, log_dir)
	env.reset()

	def check_buffers_full():
		for agent in agent_list:
			if agent.replay_buffer.num_frames() < agent.replay_buffer_capacity:
				return True
		return False

	log_interval = 5000

	while check_buffers_full():
		env.walk()
		if global_step.numpy() % log_interval == 0:
			print("Replay buffers:", "".join([str(int(agent.replay_buffer.num_frames()/agent.replay_buffer_capacity*10000)/100) + "%" for agent in agent_list]))


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
		replay_buffer_capacity=500000,
		replay_ratio=10
	)
