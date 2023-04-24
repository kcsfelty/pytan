import concurrent.futures
import random
import time
from abc import ABC
from itertools import cycle

import numpy as np
import tensorflow as tf
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import TimeStep, StepType
from tf_agents.typing import types

import reference.definitions as df
from board.board import Board
from game.handler import Handler
from game.player import Player
from reference.settings import player_count, development_card_count_per_type, resource_card_count_per_type
from reference.specs import reward_spec, action_spec, observation_spec, discount_spec, time_step_spec
from reference.state import State
from util.Dice import Dice
from util.reverse_histogram import reverse_histogram


class PyTan(PyEnvironment, ABC):
	def __init__(self, game_count=1, global_step=None, worker_count=1, log_dir="./logs", env_id=""):

		super(PyTan, self).__init__(handle_auto_reset=True)
		self.game_count = game_count
		self.global_step = global_step
		self.worker_count = worker_count
		self.env_id = env_id

		# Summaries
		self.log_dir = log_dir
		if global_step is not None:
			self.writer = tf.summary.create_file_writer(logdir=self.log_dir + "/game")

		# Environment
		self.state = State(self.game_count)
		self.player_list = self.get_player_list()
		self.board = Board(self.state, self)
		self.handler = Handler(self)
		self.dice = Dice()
		self.step_type = np.ones((player_count, self.game_count, 1), dtype=np.int32)
		self.reward = np.zeros((player_count, self.game_count, 1), dtype=np.float32)
		self.discount = np.ones((player_count, self.game_count, 1), dtype=np.float32)

		# Driver helpers
		self.last_game_at_step = 0
		self.last_game_at_time = time.perf_counter()
		self.min_turns = np.inf
		self.crash_log = [[]] * game_count
		self.num_step = [0] * game_count

		# Game logic helpers
		self.player_cycle = [None] * game_count
		self.player_order_build_phase = [[0, 1, 2] for _ in range(game_count)]
		self.player_order_build_phase_reversed = [[2, 1, 0] for _ in range(game_count)]
		self.current_player = [None] * game_count
		self.winning_player = [None] * game_count
		self.development_card_stack = [[] for _ in range(game_count)]
		self.trading_player = [None] * game_count
		self.longest_road_owner = [None] * game_count
		self.largest_army_owner = [None] * game_count
		self.player_trades_this_turn = [0] * game_count
		self.resolve_road_building_count = [0] * game_count

	@property
	def batched(self) -> bool:
		return True

	@property
	def batch_size(self) -> int:
		return self.game_count * player_count

	def reward_spec(self) -> types.NestedArraySpec:
		return reward_spec

	def action_spec(self) -> types.NestedArraySpec:
		return action_spec

	def observation_spec(self) -> types.NestedArraySpec:
		return observation_spec

	def discount_spec(self) -> types.NestedArraySpec:
		return discount_spec

	def time_step_spec(self) -> TimeStep:
		return time_step_spec

	def should_reset(self, current_time_step) -> bool:
		return False

	def get_player_list(self):
		player_list = [Player(
			index=player_index,
			game=self,
			private_state=self.state.private_state_slices[player_index],
			public_state=self.state.public_state_slices[player_index],
		) for player_index in range(player_count)]

		for player in player_list:
			for other_player in player_list:
				if player is not other_player:
					player.other_players.append(other_player)

		return player_list

	def write_episode_summary(self, game_index):
		if self.global_step:
			turn = self.state.turn_number[game_index].item()
			step = self.num_step[game_index]
			global_step = self.global_step.numpy().item()
			time_delta = time.perf_counter() - self.last_game_at_time
			self.last_game_at_time = time.perf_counter()
			step_delta = global_step - self.last_game_at_step
			self.last_game_at_step = int(global_step)
			rate = int(step_delta / time_delta)
			summary = ""
			if self.env_id:
				summary += "[env: {}] ".format(str(self.env_id).rjust(5))
			summary += "[game:{}] ".format(str(game_index).rjust(5))
			summary += "[turns:{}] ".format(str(turn).rjust(5))
			summary += "[steps:{}] ".format(str(step).rjust(6))
			summary += "[global:{}]   ".format(str(global_step).rjust(10))
			summary += "[rate:{}]   ".format(str(rate).rjust(6))
			if self.winning_player[game_index]:
				summary += str(self.winning_player[game_index].for_game(game_index))
			print(summary)

			with self.writer.as_default(step=global_step):
				tf.summary.scalar(name="turn_count", data=turn)

			for player in self.player_list:
				player.write_episode_summary(game_index)

			if turn < self.min_turns:
				if turn < 20:
					print("\n".join(self.crash_log[game_index]))

	def get_crash_log(self, player, game_index):
		player_str = player.for_game(game_index) if player else ""
		self.crash_log[game_index].append(player_str)
		return "\n".join(self.crash_log[game_index])

	def add_action_to_crash_log(self, game_index, player_index, action_handler, action_args):
		if action_handler.__name__ != "handle_no_action":
			crash_str = ""
			crash_str += str(game_index) + " "
			crash_str += str(self.state.turn_number[game_index]) + " "
			crash_str += self.player_list[player_index].for_game(game_index) + " "
			crash_str += str(self.player_list[player_index].resource_cards[game_index]) + " "
			crash_str += action_handler.__name__ + " "
			crash_str += str(action_args) if action_args is not None else "" + " "
			self.crash_log[game_index].append(crash_str)

	def _step(self, action_list: types.NestedArray) -> tuple[TimeStep, ...]:
		self.reset_time_step()
		for game_index in range(self.game_count):
			if self.winning_player[game_index]:
				self.complete_game(game_index)
			else:
				self.process_game_actions(game_index, action_list)
			if self.winning_player[game_index]:
				self.game_has_winner(game_index)
		return self.get_time_step()

	def reset_time_step(self):
		self.step_type.fill(1)
		self.reward.fill(0)
		self.discount.fill(1)

	def game_has_winner(self, game_index):
		winning_player_index = self.winning_player[game_index].index
		self.step_type[:, game_index] = StepType.LAST
		self.discount[:, game_index] = 0.
		self.reward[:, game_index] -= 1
		self.reward[winning_player_index, game_index] += 1

	def process_game_actions(self, game_index, action_list):
		with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_count) as executor:
			futures = []
			for player_index in range(player_count):
				if self.global_step:
					self.global_step.assign_add(1)
				self.num_step[game_index] += 1
				action = action_list[player_index][game_index]
				action_handler, action_args = self.handler.action_lookup[action]
				self.add_action_to_crash_log(game_index, player_index, action_handler, action_args)
				args = action_args, self.player_list[player_index], game_index
				futures.append(executor.submit(action_handler, *args))
			for future in concurrent.futures.as_completed(futures):
				future.result()

	def _reset(self):
		print("Resetting all games")
		reset_games = [x for x in range(self.game_count)]
		for game_index in reset_games:
			self.reset_game(game_index)
			self.start_game(game_index)
		return self.get_time_step()

	def complete_game(self, game_index):
		self.write_episode_summary(game_index)
		self.reset_game(game_index)
		self.step_type[:, game_index] = StepType.FIRST
		self.start_game(game_index)

	def start_game(self, game_index):
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

	def get_player_time_step(self, player_index):
		obs = self.state.for_player(player_index)
		mask = self.player_list[player_index].dynamic_mask.mask
		return TimeStep(
			step_type=self.step_type[player_index],
			reward=self.reward[player_index],
			discount=self.discount[player_index],
			observation=(obs, mask))

	def get_time_step(self):
		obs = [self.state.for_player(player_index) for player_index in range(player_count)]
		mask = [self.player_list[player_index].dynamic_mask.mask for player_index in range(player_count)]
		return TimeStep(
			step_type=self.step_type,
			reward=self.reward,
			discount=self.discount,
			observation=(np.array(obs), np.array(mask)))
