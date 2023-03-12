import time

import numpy as np
from tf_agents.environments import tf_py_environment

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


def train_eval(
		env_specs,
		log_dir="./logs/{}".format(int(time.time())),
		total_steps=9000000,
		batch_size=5000,
		train_steps=5,
	):
	steps_per_iteration = int(batch_size * train_steps)
	iterations = int(total_steps / steps_per_iteration)
	global_step = np.zeros(1, dtype=np.int64)
	eps_scaling = np.zeros(3)
	eps_scaling.fill(1)
	eps_base = 0.01

	expected_implicit_action_ratio = 0.25
	step_scaling = player_count / expected_implicit_action_ratio

	def epsilon_greedy(player_index):
		def get_epsilon_greedy():
			return eps_base  # * eps_scaling[player_index].item()
		return get_epsilon_greedy

	# random_agent_list = [RandomAgent(
	# 	env_specs=env_specs,
	# 	player_index=player_index,
	# 	log_dir=log_dir
	# ) for player_index in range(player_count)]

	agent_list = [
		FastAgent(
			player_index=0,
			batch_size=batch_size,
			log_dir=log_dir,
			replay_buffer_capacity=batch_size*train_steps,
			fc_layer_params=(2 ** 5, 2 ** 5),
			epsilon_greedy=epsilon_greedy(0),
			env_specs=env_specs),
		RandomAgent(
			env_specs=env_specs,
			player_index=1,
			log_dir=log_dir
		),
		RandomAgent(
			env_specs=env_specs,
			player_index=2,
			log_dir=log_dir
		),
		# FastAgent(
		# 	player_index=1,
		# 	batch_size=batch_size,
		# 	log_dir=log_dir,
		# 	replay_buffer_capacity=batch_size*train_steps,
		# 	fc_layer_params=(2 ** 5, 2 ** 5),
		# 	epsilon_greedy=epsilon_greedy(1),
		# 	env_specs=env_specs),
		# FastAgent(
		# 	player_index=2,
		# 	batch_size=batch_size,
		# 	log_dir=log_dir,
		# 	replay_buffer_capacity=batch_size*train_steps,
		# 	fc_layer_params=(2 ** 5, 2 ** 5),
		# 	epsilon_greedy=epsilon_greedy(2),
		# 	env_specs=env_specs)
	]

	# agent_list = [*(agent_list if agent_list else []), *random_agent_list[len(agent_list):]][:player_count]

	trainable_agents = [agent_list[0]]
	env = PyTanFast(agent_list, global_step, log_dir)

	env.reset()

	for i in range(iterations):
		print("starting iteration", i, "steps so far:", global_step.item(), str(int(global_step / total_steps * 1e2)) + "%",)
		env.run(step_limit=steps_per_iteration)
		for agent in trainable_agents:
			agent.train(train_steps)
		# if global_step >= 50000:
		# 	eps_base = 0.01
		# 	eps_scaling = eps_place_scaling[env.get_player_win_rate_order(50)]
		# print(eps_scaling)

if __name__ == "__main__":
	_env = PyTanFast()
	train_env = tf_py_environment.TFPyEnvironment(_env)
	env_specs = {
		"env_observation_spec": train_env.observation_spec()[0],
		"env_action_spec": train_env.action_spec()[0],
		"env_time_step_spec": train_env.time_step_spec(),
	}
	train_eval(env_specs)
