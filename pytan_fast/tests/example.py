from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import PolicyStep
from tf_agents.utils import common
import pytan_fast.definitions as df
from pytan_fast.game import PyTanFast
from pytan_fast.mask import Mask
import tensorflow as tf
import numpy as np
from pytan_fast.settings import player_list


class Policy:
	def __init__(self):
		pass
	def action(self, time_step):
		return PolicyStep(action=0)

class MultiAgent:
	def __repr__(self):
		return "<Player {}>".format(self.index)

	def __init__(self, index, policy, observers):
		self.index = index
		self.policy = policy
		self.turn_mask = Mask()
		self.availability_mask = Mask()
		self.observers = observers
		self.last_time_step = None
		self.last_action = None
		self.other_players = []

	def get_action(self, time_step):
		self.last_time_step = time_step
		action = self.policy.action(time_step)
		self.last_action = action
		return action


def traj_for(player_index):
	def add_traj(traj):
		pass
		# print("added a traj for", player_index, traj.action)
	return add_traj


def splitter(obs_tuple):
	obs, mask = obs_tuple
	return obs, mask

policy_list = [Policy(), Policy(), Policy()]
observer_list = [[traj_for(i)] for i in player_list]
env = PyTanFast(policy_list, observer_list, [])
train_env = tf_py_environment.TFPyEnvironment(env)
rando = RandomPyPolicy(
	time_step_spec=env.time_step_spec(),
	action_spec=env.action_spec()[0],
	observation_and_action_constraint_splitter=splitter
)
# def get_agent(player_index):

class FastAgent:
	def __init__(self, player_index):
		self.player_index = player_index

		self.train_step_counter = tf.Variable(0, dtype=tf.int64)

		self.categorical_q_net = categorical_q_network.CategoricalQNetwork(
			train_env.observation_spec()[0],
			action_spec=train_env.action_spec()[0],
			num_atoms=51,
			fc_layer_params=(1024, 1024))

		self.agent = categorical_dqn_agent.CategoricalDqnAgent(
			time_step_spec=train_env.time_step_spec(),
			action_spec=train_env.action_spec()[0],
			categorical_q_network=self.categorical_q_net,
			optimizer=tf.compat.v1.train.AdamOptimizer(),
			min_q_value=-1,
			max_q_value=1,
			n_step_update=4,
			td_errors_loss_fn=common.element_wise_squared_loss,
			gamma=0.9,
			epsilon_greedy=0.1,
			observation_and_action_constraint_splitter=splitter,
			train_step_counter=self.train_step_counter)

		self.agent.initialize()

		self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
			data_spec=self.agent.collect_data_spec,
			batch_size=1,
			max_length=10000)

		self.dataset = self.replay_buffer.as_dataset(
			num_parallel_calls=3,
			sample_batch_size=100,
			num_steps=4 + 1).prefetch(3)

		self.iterator = iter(self.dataset)

		self.writer = tf.compat.v2.summary.create_file_writer("./training/agent" + str(player_index))

		self.write_count = 0

	def write_summary(self, rewards):

		with self.writer.as_default():
			tf.summary.scalar(name="episode_rewards".format(self.player_index), data=rewards, step=self.write_count)
		self.write_count += 1

	def train(self):
		exp, _ = next(self.iterator)
		# loss = self.agent.train(exp)
		with self.writer.as_default():
			loss = self.agent.train(exp)

fast1 = FastAgent(0)
fast2 = FastAgent(1)
fast3 = FastAgent(2)

policy_list = [fast1.agent.collect_policy, fast2.agent.collect_policy, fast3.agent.collect_policy]
observer_list = [[fast1.replay_buffer.add_batch], [fast2.replay_buffer.add_batch], [fast3.replay_buffer.add_batch]]
summary_list = [fast1.write_summary, fast2.write_summary, fast3.write_summary]
env = PyTanFast(policy_list, observer_list, summary_list)


def test_disbursement(disbursement):
	print()
	print(disbursement)
	print(env.handler.check_bank_can_disburse(disbursement))


test_disbursement(np.array([
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0]
]))

test_disbursement(np.array([
	[19, 19, 19, 19, 19],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0]
]))

test_disbursement(np.array([
	[20, 20, 20, 20, 20],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0]
]))

test_disbursement(np.array([
	[1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1]
]))

test_disbursement(np.array([
	[10, 1, 1, 1, 1],
	[10, 1, 1, 1, 1],
	[1, 1, 1, 1, 1]
]))

















