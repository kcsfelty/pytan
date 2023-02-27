import random
import time
from abc import ABC
from itertools import cycle
from typing import Sequence, Callable, Any, Optional
import tensorflow as tf
import numpy as np
from tf_agents import trajectories
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.environments import PyEnvironment, tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.policies import py_policy
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.replay_buffers import TFUniformReplayBuffer, tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import TimeStep, Trajectory, PolicyStep, trajectory, StepType
from tf_agents.trajectories.policy_step import ActionType
from tf_agents.typing import types
from tf_agents.utils import common

import pytan_fast.definitions as df
from pytan_fast.board import Board
from pytan_fast.handler import Handler
from pytan_fast.mask import Mask
from pytan_fast.player import Player
from pytan_fast.settings import player_count, development_card_count_per_type, resource_card_count_per_type, player_list, victory_point_card_index
from pytan_fast.states.state import State
from util.Dice import Dice


rng = np.random.default_rng()


def reverse_histogram(hist):
	hist = np.repeat([x for x in range(len(hist))], hist).tolist()
	rng.shuffle(hist)
	return hist

class PyTanFast(PyEnvironment, ABC):
	def __init__(self, policy_list, observer_list, summary_list):
		super().__init__()
		self.state = State()
		self.board = Board(self.state, self)
		self.summary_list = summary_list
		self.player_list = None

		if not self.player_list:
			self.player_list = [Player(
				index=player_index,
				game=self,
				policy=policy_list[player_index],
				observers=observer_list[player_index],
				private_state=self.state.private_state_slices[player_index],
				public_state=self.state.public_state_slices[player_index],
			) for player_index in range(player_count)]

		for player in self.player_list:
			for other_player in self.player_list:
				if player is not other_player:
					player.other_players.append(other_player)

		self.handler = Handler(self)
		self.immediate_play = []
		# no_action_code = np.array(379, dtype=np.int16)
		# self.no_action_step = PolicyStep(action=no_action_code)
		self.dice = Dice()

		# Driver helpers
		self.num_step = None
		self.turn_limit = None
		self.step_limit = None
		self.victory_point_limit = None
		self.last_victory_points = np.zeros(player_count, dtype=np.float_)
		self.player_cycle = None
		self.current_player = None
		self.winning_player = None

		# Game logic helpers
		self.development_card_stack = []
		self.trading_player = None
		self.longest_road_owner = None
		self.largest_army_owner = None
		self.player_trades_this_turn = 0
		self.resolve_road_building_count = 0

		self._reset()

		self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=379 - 1, name='action'),

		self._observation_spec = (
			array_spec.BoundedArraySpec(shape=(len(self.state.for_player(0)),), dtype=np.int32, minimum=0, name='observation'),
			array_spec.BoundedArraySpec(shape=(379,), dtype=np.int32, minimum=0, maximum=1, name='action_mask'))

	def action_spec(self) -> types.NestedArraySpec:
		return self._action_spec

	def observation_spec(self) -> types.NestedArraySpec:
		return self._observation_spec

	def _step(self, action: types.NestedArray) -> TimeStep:
		pass

	def _reset(self):
		self.state.reset()
		self.board.reset()
		self.last_victory_points.fill(0)
		dev_card_count = sum(development_card_count_per_type)
		self.development_card_stack = reverse_histogram(development_card_count_per_type)
		self.state.game_state_slices[df.bank_development_card_count] += dev_card_count
		# self.development_card_stack = random.sample(dev_card_rev_hist, len(dev_card_rev_hist))
		self.state.game_state_slices[df.bank_resources] += resource_card_count_per_type
		self.state.game_state_slices[df.build_phase].fill(1)
		self.num_step = 0
		self.state.game_state_slices[df.vertex_open].fill(1)
		self.state.game_state_slices[df.edge_open].fill(1)
		self.winning_player = None

		for player in self.player_list:
			player.reset()

		for tile in self.board.tiles:
			tile.reset()

		for vertex in self.board.vertices:
			vertex.reset()

		for edge in self.board.edges:
			edge.reset()

	def get_observation(self, player):
		obs = np.expand_dims(self.state.for_player(player.index), axis=0)
		mask = np.expand_dims(player.dynamic_mask.mask, axis=0)
		obs = tf.convert_to_tensor(obs, dtype=tf.int32)
		mask = tf.convert_to_tensor(mask, dtype=tf.int32)
		return obs, mask

	def get_discount(self, exp_scale=2, offset=3):
		vp_list = [player.actual_victory_points for player in self.player_list]
		result = 1. - ((max(vp_list) - offset) / self.victory_point_limit) ** exp_scale
		result = np.expand_dims(result, axis=0)
		result = tf.convert_to_tensor(result, dtype=tf.float32)
		return result

	def get_step_type(self):
		if self.num_step == 0:
			step_type = StepType.FIRST
		elif self.check_turn_limit() or self.check_victory_points() or self.check_step_limit():
			step_type = StepType.LAST
		else:
			step_type = StepType.MID
		step_type = np.expand_dims(step_type, axis=0)
		step_type = tf.convert_to_tensor(step_type)
		return step_type

	def get_reward(self, player):
		result = float(player.actual_victory_points - self.last_victory_points[player.index])
		self.last_victory_points[player.index] = player.actual_victory_points
		if self.winning_player:
			result += 2 if player == self.winning_player else -1
		expected_steps = 700
		overall_point_reduction = 2
		result -= overall_point_reduction / expected_steps
		player.episode_rewards += result
		result /= 2
		# if self.num_step % 250 == 0:
		# 	print(player.index, result, player.episode_rewards)
		result = np.array(result, dtype=np.double)
		result = np.expand_dims(result, axis=0)
		result = tf.convert_to_tensor(result, dtype=tf.float32)
		return result

	def get_time_step(self, player):
		return TimeStep(self.get_step_type(), self.get_reward(player), self.get_discount(), self.get_observation(player))

	def decide(self, active_player):
		for player in self.player_list:
			player.start_trajectory()
		self.handler.handle_action(active_player.last_action, active_player)
		for player in self.player_list:
			player.end_trajectory()
		self.check_victory_points()

	def build_phase(self, player_order):
		for build_player in player_order:
			build_player.dynamic_mask.only(df.place_settlement)
			np.logical_and(build_player.dynamic_mask.place_settlement, self.state.game_state_slices[df.vertex_open], out=build_player.dynamic_mask.place_settlement)
			self.decide(build_player)
			self.decide(build_player)
			build_player.dynamic_mask.only(df.no_action)

	def run(self, turn_limit=99, step_limit=1000, victory_point_limit=10, episode_limit=15):
		self.turn_limit = turn_limit
		self.step_limit = step_limit
		self.victory_point_limit = victory_point_limit
		for game_index in range(episode_limit):
			self._reset()
			print("starting game", game_index)
			start = time.perf_counter()
			player_order = random.sample(self.player_list, len(self.player_list))
			# print("player_order", [player.index for player in player_order])
			self.player_cycle = cycle([x for x in player_order])

			self.build_phase(player_order)
			player_order.reverse()
			self.state.game_state_slices[df.build_phase_reversed].fill(1)
			self.build_phase(player_order)
			self.state.game_state_slices[df.build_phase].fill(0)
			self.state.game_state_slices[df.build_phase_reversed].fill(0)
			print("---------------------------------------------")
			self.current_player = next(self.player_cycle)
			self.current_player.dynamic_mask.only(df.roll_dice)

			while self.check_turn_limit() and self.check_victory_points() and self.check_step_limit():
				self.current_player = self.immediate_play.pop(0) if self.immediate_play else self.current_player
				self.decide(self.current_player)
				self.num_step += 1

			for writer, player in zip(self.summary_list, self.player_list):
				writer(player.episode_rewards)

			end = time.perf_counter()
			print("Game completed", self.last_victory_points, "took", end - start, "steps", self.num_step, "turns", self.state.game_state_slices[df.turn_number], "steps/s", self.num_step / (end - start))
			for player in self.player_list:
				print(player.episode_rewards, player)
			print("=============================================")

	def check_turn_limit(self):
		if self.turn_limit:
			return self.state.game_state_slices[df.turn_number] < self.turn_limit
		return True

	def check_victory_points(self):
		for player in self.player_list:
			if player.actual_victory_points >= self.victory_point_limit:
				if player == self.current_player:
					self.winning_player = player
				return False
		return True

	def check_step_limit(self):
		if self.step_limit:
			return self.num_step < self.step_limit
		return True


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


def train_eval(
		batch_size=64,
		n_step_update=1,
		replay_buffer_capacity=1000,
	):

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
	train_env = tf_py_environment.TFPyEnvironment(env)

	# run = common.function(env.run)

	iterations = 200
	train_steps = 5
	for i in range(iterations):
		print("starting iteration", i)
		env.run(episode_limit=15*train_steps)
		print("training")
		for _ in range(train_steps):
			fast1.train()
			fast2.train()
			fast3.train()
	# end = time.perf_counter()
	# print("Game took", end - start)

	# env.player_list[0].dynamic_mask.place_settlement

if __name__ == '__main__':

	train_eval()




















