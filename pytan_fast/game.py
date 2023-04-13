import concurrent.futures
import random
import time
from abc import ABC
from itertools import cycle
from typing import Tuple

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

action_count = 379
observation_count = 1402


def reverse_histogram(hist):
	hist = np.repeat([x for x in range(len(hist))], hist).tolist()
	rng.shuffle(hist)
	return hist


class PyTanFast(PyEnvironment, ABC):
	def __init__(self, game_count=1, global_step=None, log_dir="./logs"):

		super(PyTanFast, self).__init__(
			handle_auto_reset=True
		)
		self.game_count = game_count

		# Summaries
		self.log_dir = log_dir + "/game"
		self.episode_number = 0
		self.writer = tf.summary.create_file_writer(logdir=self.log_dir)

		# Environment
		self.step_type = np.ones((player_count, self.game_count), dtype=np.int32)
		self.reward = np.zeros((player_count, self.game_count), dtype=np.float32)
		self.discount = np.ones((player_count, self.game_count), dtype=np.float32)
		self.global_step = global_step
		self.state = State(self.game_count, player_count)
		self.board = Board(self.state, self)
		self.player_list = None

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
		self.dice = Dice()

		# Driver helpers
		self.last_game_at_step = 0
		self.last_game_at_time = time.perf_counter()
		self.min_turns = np.inf
		self.crash_log = [[]] * game_count
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

		self._action_spec = (BoundedArraySpec(
			shape=(game_count,),
			dtype=np.int32,
			minimum=0,
			maximum=action_count - 1,
			name='action'),) * player_count

		self._discount_spec = BoundedArraySpec(
			shape=(game_count,),
			dtype=np.float32,
			minimum=0.,
			maximum=1.,
			name='discount')
		self._observation_spec = (
			BoundedArraySpec(
				shape=(game_count, observation_count,),
				dtype=np.int32,
				minimum=0,
				maximum=127,
				name='observation'),
			BoundedArraySpec(
				shape=(game_count, action_count,),
				dtype=np.int32,
				minimum=0,
				maximum=1,
				name='action_mask'))
		self._reward_spec = BoundedArraySpec(
			shape=(game_count,),
			dtype=np.float32,
			minimum=-1.,
			maximum=1.,
			name='reward')
		self._step_type_spec = ArraySpec(
			shape=(game_count,),
			dtype=np.int32,
			name='step_type')
		self._time_step_spec = TimeStep(
			step_type=self._step_type_spec,
			reward=self._reward_spec,
			discount=self._discount_spec,
			observation=self._observation_spec
		) * player_count

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

	def should_reset(self, current_time_step) -> bool:
		return False

	def write_episode_summary(self, game_index):
		turn = self.state.turn_number[game_index].item()
		step = self.num_step[game_index]
		global_step = self.global_step.numpy().item()
		time_delta = time.perf_counter() - self.last_game_at_time
		self.last_game_at_time = time.perf_counter()
		step_delta = global_step - self.last_game_at_step
		self.last_game_at_step = int(global_step)
		rate = int(step_delta / time_delta)
		summary = ""
		summary += "[env:{}] ".format(str(game_index).rjust(5))
		summary += "[turns:{}] ".format(str(turn).rjust(5))
		summary += "[steps:{}] ".format(str(step).rjust(6))
		summary += "[global:{}]   ".format(str(global_step).rjust(10))
		summary += "[rate:{}]   ".format(str(rate).rjust(6))
		if self.winning_player[game_index]:
			summary += str(self.winning_player[game_index].for_game(game_index))
		print(summary)

		with self.writer.as_default(step=global_step):
			tf.summary.scalar(name="turn_count", data=turn)

		if turn < self.min_turns:
			if turn < 20:
				print(self.get_crash_log(None, game_index, add_state=False))

	def add_state_to_crash_log(self, player, game_index):
		pass
		# for term in self.state.game_state_slices:
		# 	term_str = str(term) + ", " + str(self.state.game_state_slices[term][game_index])
		# 	self.crash_log[game_index].append(term_str)
		# for term in player.public_state:
		# 	term_str = str(term) + ", " + str(player.public_state[term][game_index])
		# 	self.crash_log[game_index].append(term_str)
		# for term in player.private_state:
		# 	term_str = str(term) + ", " + str(player.private_state[term][game_index])
		# 	self.crash_log[game_index].append(term_str)

	def get_crash_log(self, player, game_index, add_state=True):
		if add_state:
			self.add_state_to_crash_log(player or None, game_index)
			self.crash_log[game_index].append(player.for_game(game_index))
		return "\n".join(self.crash_log[game_index])

	def add_action_to_crash_log(self, game_index, player_index, action_handler, action_args):
		if action_handler.__name__ is not "handle_no_action":
			crash_str = ""
			crash_str += str(game_index) + " "
			crash_str += str(self.state.turn_number[game_index]) + " "
			crash_str += self.player_list[player_index].for_game(game_index) + " "
			crash_str += str(self.player_list[player_index].resource_cards[game_index]) + " "
			crash_str += action_handler.__name__ + " "
			crash_str += str(action_args) if action_args is not None else "" + " "
			self.crash_log[game_index].append(crash_str)

	def _step(self, action_list: types.NestedArray) -> tuple[TimeStep, ...]:
		self.step_type = np.ones((player_count, self.game_count), dtype=np.int32)
		self.reward = np.zeros((player_count, self.game_count), dtype=np.float32)
		self.discount = np.ones((player_count, self.game_count), dtype=np.float32)
		for game_index in range(self.game_count):
			if self.winning_player[game_index]:
				self.write_episode_summary(game_index)
				self.reset_game(game_index)
				self.step_type[:, game_index] = StepType.FIRST
			elif self.state.turn_number[game_index].item() >= 1000:
				self.reset_game(game_index)
				self.step_type[:, game_index] = StepType.FIRST
				self.reward[:, game_index] = -1
			else:
				with concurrent.futures.ThreadPoolExecutor(max_workers=2 ** 8) as executor:
					futures = []
					for player_index in range(player_count):
						self.global_step.assign_add(1)
						self.num_step[game_index] += 1
						action = action_list[player_index][game_index]
						action_handler, action_args = self.handler.action_lookup[action]
						self.add_action_to_crash_log(game_index, player_index, action_handler, action_args)
						args = action_args, self.player_list[player_index], game_index
						futures.append(executor.submit(action_handler, *args))
						# action_handler(*args)
					for future in concurrent.futures.as_completed(futures):
						future.result()
			if self.winning_player[game_index]:
				winning_player_index = self.winning_player[game_index].index
				self.step_type[:, game_index] = StepType.LAST
				self.discount[:, game_index] = 0.
				self.reward[:, game_index] = -1
				self.reward[winning_player_index, game_index] = 1

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
		self.crash_log[game_index] = [""]

		for player in self.player_list: player.reset(game_index)
		for tile in self.board.tiles: tile.reset(game_index)
		for vertex in self.board.vertices: vertex.reset(game_index)
		for edge in self.board.edges: edge.reset(game_index)

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
		first_player.current_player[game_index].fill(True)
		self.current_player[game_index] = first_player
		np.logical_and(
			first_player.dynamic_mask.place_settlement[game_index],
			self.state.vertex_open[game_index],
			out=first_player.dynamic_mask.place_settlement[game_index])

	def get_observation(self):
		obs_list = [self.state.for_player(player_index) for player_index in range(player_count)]
		mask_list = [self.player_list[player_index].dynamic_mask.mask for player_index in range(player_count)]
		observation = [(np.array(obs), np.array(mask)) for obs, mask in zip(obs_list, mask_list)]
		return observation

	def get_player_time_step(self, player_index):
		obs = self.state.for_player(player_index)
		mask = self.player_list[player_index].dynamic_mask.mask
		return TimeStep(
			step_type=self.step_type[player_index],
			reward=self.reward[player_index],
			discount=self.discount[player_index],
			observation=(obs, mask))

	def get_time_step(self):
		return tuple(self.get_player_time_step(player_index) for player_index in range(3))
