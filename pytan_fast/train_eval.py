import os
import time

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from pytan_fast.agent import FastAgent
from pytan_fast.game import PyTanFast
from pytan_fast.random_agent import RandomAgent
from pytan_fast.settings import player_count

eps_start = 1.
eps_end = 0.0875
eps_steps = 1
eps_range = eps_start - eps_end
eps_delta = eps_range / eps_steps

eps_base = 0.1
eps_place_scaling = np.array([0.0, 1.0, 2.0])


def init_agents(
		agent_list,
		global_step,
		log_dir,
		batch_size,
		init_steps,
		train_steps,
	):
	random_agent_list = [
		RandomAgent(
			env_specs=env_specs,
			player_index=9 - i,
			log_dir=log_dir
		)
		for i in range(len(agent_list))]
	init_steps_per_iteration = int(batch_size * train_steps)
	init_iterations = int(init_steps / init_steps_per_iteration)
	init_agent_env_list = []
	for j in range(len(agent_list)):
		init_agent_list = [agent_list[i] if i == j else random_agent_list[i] for i in range(len(agent_list))]
		init_agent_env_list.append(PyTanFast(init_agent_list, global_step, log_dir))
	for init_env in init_agent_env_list:
		init_env.reset()
	for init_index in range(player_count):
		for iteration in range(init_iterations):
			init_agent_env_list[init_index].run(step_limit=init_steps_per_iteration)
			agent_list[init_index].train(train_steps)
	# for iteration in range(init_iterations):
	# 	for init_env in init_agent_env_list:
	# 		init_env.run(step_limit=init_steps_per_iteration)
	# 	for agent in agent_list:
	# 		agent.train(train_steps)


def train_eval(
		env_specs,
		log_dir="./logs/{}".format(int(time.time())),
		total_steps=4000000,
		init_steps=None,
		batch_size=32,
		train_steps=1,
		replay_buffer_scaling=5,
		warmup_steps=3000,
	):
	steps_per_iteration = int(batch_size * train_steps)
	iterations = int(total_steps / steps_per_iteration)
	global_step = np.zeros(1, dtype=np.int64)
	eps_scaling = np.zeros(3)
	eps_scaling.fill(1)
	eps_base = 1.

	expected_implicit_action_ratio = 0.25
	step_scaling = player_count / expected_implicit_action_ratio

	def epsilon_greedy(player_index):
		def get_epsilon_greedy():
			return eps_base * eps_scaling[player_index].item()
		return get_epsilon_greedy

	# random_agent_list = [RandomAgent(
	# 	env_specs=env_specs,
	# 	player_index=player_index,
	# 	log_dir=log_dir
	# ) for player_index in range(player_count)]
	# agent_list = [
	# 	FastAgent(
	# 		player_index=0,
	# 		batch_size=batch_size,
	# 		log_dir=log_dir,
	# 		replay_buffer_capacity=batch_size*train_steps,
	# 		fc_layer_params=(2 ** 5, 2 ** 5),
	# 		epsilon_greedy=epsilon_greedy(0),
	# 		env_specs=env_specs),
	# 	FastAgent(
	# 		player_index=1,
	# 		batch_size=batch_size,
	# 		log_dir=log_dir,
	# 		replay_buffer_capacity=batch_size*train_steps,
	# 		fc_layer_params=(2 ** 5, 2 ** 5),
	# 		epsilon_greedy=epsilon_greedy(1),
	# 		env_specs=env_specs),
	# 	FastAgent(
	# 		player_index=2,
	# 		batch_size=batch_size,
	# 		log_dir=log_dir,
	# 		replay_buffer_capacity=batch_size*train_steps,
	# 		fc_layer_params=(2 ** 5, 2 ** 5),
	# 		epsilon_greedy=epsilon_greedy(2),
	# 		env_specs=env_specs)
	# ]
	#
	# random_agent_list = [
	# 	RandomAgent(
	# 		env_specs=env_specs,
	# 		player_index=7,
	# 		log_dir=log_dir
	# 	),
	# 	RandomAgent(
	# 		env_specs=env_specs,
	# 		player_index=8,
	# 		log_dir=log_dir
	# 	),
	# 	RandomAgent(
	# 		env_specs=env_specs,
	# 		player_index=9,
	# 		log_dir=log_dir
	# 	),
	# ]

	agent_list = [
		FastAgent(
			player_index=i,
			batch_size=batch_size,
			log_dir=log_dir,
			replay_buffer_capacity=100000,
			fc_layer_params=(2 ** 7, 2 ** 6),
			learning_rate=0.001,
			# learning_rate=1/(batch_size * train_steps * 2),
			epsilon_greedy=epsilon_greedy(i),
			n_step_update=5,
			# epsilon_greedy=eps_base,
			env_specs=env_specs)
		for i in range(player_count)]

	if init_steps:
		init_agents(
			agent_list=agent_list,
			global_step=global_step,
			log_dir=log_dir,
			batch_size=batch_size,
			init_steps=init_steps,
			train_steps=train_steps)

	env = PyTanFast(agent_list, global_step, log_dir)
	env.reset()
	if warmup_steps:
		env.run(step_limit=warmup_steps * step_scaling)
	for i in range(iterations):
		start = time.perf_counter()
		env.run(step_limit=steps_per_iteration * step_scaling)
		for agent in agent_list:
			agent.train(train_steps)
		end = time.perf_counter()
		print("Iteration took", int(end - start), "s", "steps/sec:", int((steps_per_iteration * step_scaling) / (end - start)))
		if global_step >= 50000:
			eps_base = 0.05
			eps_scaling = eps_place_scaling[env.get_player_win_rate_order(50)]
		# print(eps_scaling)


def build_random_experience_dataset(
		env_specs,
		log_dir="./logs/{}".format(int(time.time())),
		rb_dir="./rb/{}".format(int(time.time())),
		total_steps=1000000,
		rb_capacity=1000000,
		n_step_update=12,
		batch_size=64,
		save_interval=10000,
		step_scaling=1 / 0.35
	):

	init_agent = FastAgent(
		player_index=0,
		batch_size=batch_size,
		log_dir=log_dir,
		replay_buffer_capacity=rb_capacity,
		n_step_update=n_step_update,
		env_specs=env_specs)

	replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
		data_spec=init_agent.agent.collect_data_spec,
		batch_size=1,
		max_length=rb_capacity)

	# dataset = replay_buffer.as_dataset(
	# 	sample_batch_size=batch_size,
	# 	num_steps=n_step_update + 1)

	random_agent_list = [RandomAgent(
		env_specs=env_specs,
		player_index=player_index,
		log_dir=log_dir,
		observers=[replay_buffer.add_batch]
	) for player_index in range(player_count)]

	global_step = np.zeros(1, dtype=np.int64)
	env = PyTanFast(random_agent_list, global_step, log_dir)
	env.reset()
	checkpoint_args = {
		"directory": rb_dir,
		"max_to_keep": 1000
	}
	rb_checkpointer = common.Checkpointer(
		ckpt_dir=os.path.join(rb_dir, 'replay_buffer'),
		max_to_keep=1,
		replay_buffer=replay_buffer)

	while global_step < total_steps:
		print(replay_buffer.num_frames())
		start = time.perf_counter()
		env.run(save_interval * step_scaling)
		end = time.perf_counter()
		interval = end - start
		step_rate = (save_interval * step_scaling) / interval
		print("Iteration took", int(interval), "s", "steps/sec:", int(step_rate))
		rb_checkpointer.save(global_step=tf.convert_to_tensor(global_step, dtype=tf.int32))
		# new_data = dataset.take(100000)
		# new_data.save(path=rb_dir, checkpoint_args=checkpoint_args)



if __name__ == "__main__":
	_env = PyTanFast()
	train_env = tf_py_environment.TFPyEnvironment(_env)
	env_specs = {
		"env_observation_spec": train_env.observation_spec()[0],
		"env_action_spec": train_env.action_spec()[0],
		"env_time_step_spec": train_env.time_step_spec(),
	}
	# train_eval(env_specs)
	build_random_experience_dataset(env_specs)
