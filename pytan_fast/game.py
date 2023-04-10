import random
from abc import ABC
from itertools import cycle

import numpy as np
import tensorflow as tf
from tf_agents.environments import PyEnvironment
from tf_agents.specs import BoundedArraySpec, ArraySpec
from tf_agents.trajectories import TimeStep, StepType
from tf_agents.typing import types

import pytan_fast.definitions as df
from pytan_fast.board import Board
from pytan_fast.handler import Handler
from pytan_fast.player import Player
from pytan_fast.settings import player_count, development_card_count_per_type, resource_card_count_per_type
from pytan_fast.states.state import State
from util.Dice import Dice

rng = np.random.default_rng()

expected_steps = 1800
overall_point_reduction = 0
time_drain_reward = overall_point_reduction / expected_steps

action_count = 379
observation_count = 1402


def reverse_histogram(hist):
	hist = np.repeat([x for x in range(len(hist))], hist).tolist()
	rng.shuffle(hist)
	return hist


class PyTanFast(PyEnvironment, ABC):
	def __init__(self,  game_count=1, global_step=None, log_dir="./logs", victory_point_limit=10, condensed_state=False, env_index=None, eval=False, lock=None):

		super(PyTanFast, self).__init__(
			handle_auto_reset=True
		)
		self.game_count = game_count

		# Summaries
		self.log_dir = log_dir + "/game"
		self.episode_number = 0
		# self.writer = tf.compat.v2.summary.create_file_writer(self.log_dir)
		self.writer = tf.summary.create_file_writer(logdir=self.log_dir)

		# Environment
		self.step_type = np.ones((player_count, self.game_count), dtype=np.int32)
		self.reward = np.zeros((player_count, self.game_count), dtype=np.float32)
		self.discount = np.ones((player_count, self.game_count), dtype=np.float32)
		self.env_index = env_index
		self.eval = eval
		self.lock = lock
		self.global_step = global_step
		self.condensed_state = condensed_state
		self.state = State(self.game_count, player_count)
		self.board = Board(self.state, self)
		self.player_list = None
		self.victory_point_limit = victory_point_limit

		if not self.player_list:
			self.player_list = [Player(
				index=player_index,
				game=self,
				private_state=self.state.private_state_slices[player_index],
				public_state=self.state.public_state_slices[player_index],
			) for player_index in range(player_count)]

			for player in self.player_list:
				for other_player in self.player_list:
					if player is not other_player:
						player.other_players.append(other_player)

		self.handler = Handler(self)
		self.immediate_play = [[]] * game_count
		self.dice = Dice()

		# Driver helpers
		self.num_step = [0] * game_count
		self.player_cycle = [None] * game_count
		self.player_order_build_phase = [[0, 1, 2] for _ in range(game_count)]
		self.player_order_build_phase_reversed = [[2, 1, 0] for _ in range(game_count)]
		self.current_player = [None] * game_count
		self.winning_player = [None] * game_count

		# Game logic helpers
		self.development_card_stack = [[] for _ in range(game_count)]
		self.trading_player = [None] * game_count
		self.longest_road_owner = [None] * game_count
		self.largest_army_owner = [None] * game_count
		self.player_trades_this_turn = [0] * game_count
		self.resolve_road_building_count = [0] * game_count

		batch_size = player_count * game_count

		self._discount_spec = BoundedArraySpec(shape=(batch_size,), dtype=np.float32, minimum=0., maximum=1., name='discount')
		self._action_spec = BoundedArraySpec(shape=(batch_size,), dtype=np.int32, minimum=0, maximum=action_count - 1, name='action')
		self._observation_spec = (
			BoundedArraySpec(shape=(batch_size, observation_count,), dtype=np.int32, minimum=0, maximum=128, name='observation'),
			BoundedArraySpec(shape=(batch_size, action_count,), dtype=np.int32, minimum=0, maximum=1, name='action_mask')
		)
		self._reward_spec = BoundedArraySpec(shape=(batch_size,), dtype=np.float32, minimum=-1., maximum=1., name='reward')
		self._step_type_spec = ArraySpec((batch_size,), np.int32, name='step_type')

		self._time_step_spec = TimeStep(
			step_type=self._step_type_spec,
			reward=self._reward_spec,
			discount=self._discount_spec,
			observation=self._observation_spec
		)

	@property
	def batched(self) -> bool:
		return True

	@property
	def batch_size(self) -> int:
		return self.game_count * player_count

	def reward_spec(self) -> types.NestedArraySpec:
		return self._reward_spec

	def action_spec(self) -> types.NestedArraySpec:
		return self._action_spec

	def observation_spec(self) -> types.NestedArraySpec:
		return self._observation_spec

	def discount_spec(self) -> types.NestedArraySpec:
		return self._discount_spec

	def time_step_spec(self) -> TimeStep:
		return self._time_step_spec

	def write_episode_summary(self, game_index):
		turn = self.state.turn_number[game_index].item()
		step = self.num_step[game_index]
		global_step = self.global_step.numpy().item()
		summary = ""
		summary += "[env{}]".format(str(game_index).rjust(5))
		summary += "Game finished"
		summary += str(turn).rjust(5)
		summary += " turns, "
		summary += str(step).rjust(6)
		summary += ", "
		summary += str(self.winning_player[game_index])
		summary += ", "
		summary += str(global_step)
		print(summary)

		with self.writer.as_default(step=global_step):
			tf.summary.scalar(name="turn_count", data=turn)

	def _step(self, action_list: types.NestedArray) -> TimeStep:
		self.step_type = np.ones((player_count, self.game_count), dtype=np.int32)
		self.reward = np.zeros((player_count, self.game_count), dtype=np.float32)
		self.discount = np.ones((player_count, self.game_count), dtype=np.float32)
		for game_index in range(self.game_count):
			if self.winning_player[game_index]:
				self.write_episode_summary(game_index)
				self.reset_game(game_index)
				self.step_type[:, game_index] = 0
			else:
				for player_index in range(player_count):
					self.global_step.assign_add(1)
					self.num_step[game_index] += 1
					action = action_list[player_index][game_index]
					action_handler, action_args = self.handler.action_lookup[action]
					action_handler(action_args, self.player_list[player_index], game_index)

			if self.winning_player[game_index]:
				winning_player_index = self.winning_player[game_index].index
				self.step_type[:, game_index] = StepType.LAST
				self.discount[:, game_index] = 0.
				self.reward[:, game_index] = -1
				self.reward[winning_player_index, game_index] = 1

		self.writer.flush()
		return self.get_time_step()

	def _reset(self):
		print("Resetting all games")
		reset_games = [x for x in range(self.game_count)]
		for game_index in reset_games:
			self.reset_game(game_index)
		return self.get_time_step()

	def reset_game(self, game_index):
		self.state.reset(game_index)
		self.board.reset(game_index)
		self.development_card_stack[game_index] = reverse_histogram(development_card_count_per_type)
		self.state.bank_development_card_count[game_index] += sum(development_card_count_per_type)
		self.state.bank_resources[game_index] += resource_card_count_per_type
		self.state.build_phase[game_index].fill(1)
		self.num_step[game_index] = 0
		self.state.vertex_open[game_index].fill(1)
		self.state.edge_open[game_index].fill(1)
		self.player_cycle[game_index] = None
		self.current_player[game_index] = None
		self.winning_player[game_index] = None
		self.trading_player[game_index] = None
		self.longest_road_owner[game_index] = None
		self.largest_army_owner[game_index] = None
		self.player_trades_this_turn[game_index] = 0
		self.resolve_road_building_count[game_index] = 0

		for player in self.player_list: player.reset(game_index)
		for tile in self.board.tiles: tile.reset(game_index)
		for vertex in self.board.vertices: vertex.reset(game_index)
		for edge in self.board.edges: edge.reset(game_index)

		# Get the game ready
		for player in self.player_list:
			player.dynamic_mask.only(df.no_action, game_index)

		player_index_list = [x for x in range(player_count)]
		player_order = random.sample(player_index_list, len(player_index_list))
		self.player_order_build_phase[game_index] = [x for x in player_order]
		player_order.reverse()
		self.player_order_build_phase_reversed[game_index] = [x for x in player_order]
		self.player_cycle[game_index] = cycle([x for x in player_order])
		first_player_index = self.player_order_build_phase[game_index].pop(0)
		first_player = self.player_list[first_player_index]
		first_player.dynamic_mask.only(df.place_settlement, game_index)
		np.logical_and(
			first_player.dynamic_mask.place_settlement[game_index],
			self.state.vertex_open[game_index],
			out=first_player.dynamic_mask.place_settlement[game_index])

	def get_observation(self):
		obs = [self.state.for_player(player_index) for player_index in range(player_count)]
		obs = np.array(obs)
		obs.shape = (self.batch_size, -1)
		mask = [self.player_list[player_index].dynamic_mask.mask for player_index in range(player_count)]
		mask = np.array(mask)
		mask.shape = (self.batch_size, -1)
		return obs, mask

	def get_time_step(self):
		return TimeStep(
				self.step_type.flatten(),
				self.reward.flatten(),
				self.discount.flatten(),
				self.get_observation()
			)
