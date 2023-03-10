import cProfile
import time

import numpy as np
from tf_agents.environments import tf_py_environment, tf_environment
from tf_agents.policies import PolicySaver
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.trajectories import PolicyStep

from pytan_fast.agent import FastAgent
from pytan_fast.game import PyTanFast
from pytan_fast.settings import player_list


def splitter(obs_tuple):
	obs, mask = obs_tuple
	return obs, mask

def traj_for(player_index):
	def add_traj(traj):
		pass
	return add_traj

class Policy:
	def __init__(self):
		pass

	def action(self, time_step):
		return PolicyStep(action=0)

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
		batch_size=1000,
		train_steps=5,
	):
	steps_per_iteration = int(batch_size * train_steps)
	iterations = int(total_steps / steps_per_iteration)
	global_step = np.zeros(1, dtype=np.int64)
	eps_scaling = np.zeros(3)
	eps_scaling.fill(1)
	eps_base = 0.5
	def epsilon_greedy(player_index):
		def get_epsilon_greedy():
			return eps_base * eps_scaling[player_index].item()
		return get_epsilon_greedy


	agent_list = [
		FastAgent(
			player_index=0,
			batch_size=batch_size,
			log_dir=log_dir,
			replay_buffer_capacity=batch_size*train_steps,
			fc_layer_params=(2 ** 5, 2 ** 5),
			epsilon_greedy=epsilon_greedy(0),
			env_specs=env_specs),
		FastAgent(
			player_index=1,
			batch_size=batch_size,
			log_dir=log_dir,
			replay_buffer_capacity=batch_size*train_steps,
			fc_layer_params=(2 ** 5, 2 ** 5),
			epsilon_greedy=epsilon_greedy(1),
			env_specs=env_specs),
		FastAgent(
			player_index=2,
			batch_size=batch_size,
			log_dir=log_dir,
			replay_buffer_capacity=batch_size*train_steps,
			fc_layer_params=(2 ** 5, 2 ** 5),
			epsilon_greedy=epsilon_greedy(2),
			env_specs=env_specs)]

	policy_list = [agent.agent.collect_policy for agent in agent_list]
	observer_list = [[agent.replay_buffer.add_batch] for agent in agent_list]
	summary_list = [agent.write_summary for agent in agent_list]
	env = PyTanFast(policy_list, observer_list, summary_list, global_step, log_dir)

	for i in range(iterations):
		print("starting iteration", i, "steps so far:", global_step.item(), str(int(global_step / total_steps * 1e2)) + "%",)
		env.run(step_limit=steps_per_iteration)
		for agent, a in zip(agent_list, range(len(agent_list))):
			agent.train(train_steps)
		if global_step >= 50000:
			eps_base = 0.01
			eps_scaling = eps_place_scaling[env.get_player_win_rate_order(50)]
		print(eps_scaling)

def main():
	policy_list = [Policy(), Policy(), Policy()]
	observer_list = [[traj_for(i)] for i in player_list]
	env = PyTanFast(policy_list, observer_list, [], np.zeros(1))
	train_env = tf_py_environment.TFPyEnvironment(env)
	env_specs = {
		"env_observation_spec": train_env.observation_spec()[0],
		"env_action_spec": train_env.action_spec()[0],
		"env_time_step_spec": train_env.time_step_spec(),
	}
	train_eval(env_specs)

if __name__ == "__main__":
	main()
