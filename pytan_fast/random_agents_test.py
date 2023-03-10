import time

import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
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
	global_step = np.zeros(1, dtype=np.int64)

	policy_list = [RandomTFPolicy(
		time_step_spec=env_specs["env_time_step_spec"],
		action_spec=env_specs["env_action_spec"],
		observation_and_action_constraint_splitter=splitter,)] * 3

	observer_list = [[]] * 3
	summary_list = None
	env = PyTanFast(policy_list, observer_list, summary_list, global_step, log_dir)

	print("Testing random agents")
	env.run(step_limit=total_steps)

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
