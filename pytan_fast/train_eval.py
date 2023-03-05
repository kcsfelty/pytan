import cProfile
import numpy as np
from tf_agents.environments import tf_py_environment, tf_environment
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


def train_eval(
		env_specs,
		total_steps=8000000,
		batch_size=12000,
		train_steps=3,
	):
	steps_per_iteration = int(batch_size * train_steps)
	iterations = int(total_steps / steps_per_iteration)
	global_step = np.zeros(1, dtype=np.int64)
	agent_list = [
		FastAgent(
			player_index=0,
			batch_size=batch_size,
			replay_buffer_capacity=batch_size*train_steps,
			fc_layer_params=(2 ** 5, 2 ** 5),
			env_specs=env_specs),
		FastAgent(
			player_index=1,
			batch_size=batch_size,
			replay_buffer_capacity=batch_size*train_steps,
			fc_layer_params=(2 ** 6, 2 ** 6),
			env_specs=env_specs),
		FastAgent(
			player_index=2,
			batch_size=batch_size,
			replay_buffer_capacity=batch_size*train_steps,
			fc_layer_params=(2 ** 7, 2 ** 7),
			env_specs=env_specs)]

	policy_list = [agent.agent.collect_policy for agent in agent_list]
	observer_list = [[agent.replay_buffer.add_batch] for agent in agent_list]
	summary_list = [agent.write_summary for agent in agent_list]
	env = PyTanFast(policy_list, observer_list, summary_list, global_step)
	# train_env = tf_py_environment.TFPyEnvironment(env)

	print("starting run")
	for i in range(iterations):
		print("starting iteration", i, "steps so far:", global_step)
		env.run(step_limit=steps_per_iteration)
		print("training")
		for agent, a in zip(agent_list, range(len(agent_list))):
			print("Agent{} losses:".format(str(a)))
			for j in range(train_steps):
				print("\t", j, agent.train())


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
