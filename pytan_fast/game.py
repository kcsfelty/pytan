import random
import random
import time
from abc import ABC
from itertools import cycle

import numpy as np
import tensorflow as tf
from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import TimeStep, StepType
from tf_agents.typing import types

import pytan_fast.definitions as df
from pytan_fast.board import Board
from pytan_fast.handler import Handler
from pytan_fast.player import Player
from pytan_fast.settings import player_count, development_card_count_per_type, resource_card_count_per_type, knight_index
from pytan_fast.states.state import State
from util.Dice import Dice

rng = np.random.default_rng()

expected_steps = 9999
overall_point_reduction = 10
time_drain_reward = overall_point_reduction / expected_steps


def reverse_histogram(hist):
	hist = np.repeat([x for x in range(len(hist))], hist).tolist()
	rng.shuffle(hist)
	return hist


first_step_type = tf.convert_to_tensor(np.expand_dims(StepType.FIRST, axis=0))
mid_step_type = tf.convert_to_tensor(np.expand_dims(StepType.MID, axis=0))
last_step_type = tf.convert_to_tensor(np.expand_dims(StepType.LAST, axis=0))


class PyTanFast(PyEnvironment, ABC):
	def __init__(self, policy_list, observer_list, summary_list, global_step, log_dir="./training", victory_point_limit=10):
		super().__init__()
		self.global_step = global_step
		self.state = State()
		self.board = Board(self.state, self)
		self.summary_list = summary_list
		self.player_list = None
		self.victory_point_limit = victory_point_limit

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
		self.dice = Dice()

		# Driver helpers
		self.num_step = None
		# self.turn_limit = None
		self.step_limit = None
		self.last_victory_points = np.zeros(player_count, dtype=np.float_)
		self.player_cycle = None
		self.current_player = None
		self.winning_player = None
		self.current_time_step_type = first_step_type
		self.total_steps = 0
		self.episode_start = None

		# Game logic helpers
		self.development_card_stack = []
		self.trading_player = None
		self.longest_road_owner = None
		self.largest_army_owner = None
		self.player_trades_this_turn = 0
		self.resolve_road_building_count = 0
		self.max_victory_points = 0

		# Summaries
		self.log_dir = log_dir + "/game"
		self.episode_number = 0
		self.writer = tf.compat.v2.summary.create_file_writer(self.log_dir)

		self._reset()

		self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=379 - 1, name='action'),

		self._observation_spec = (
			array_spec.BoundedArraySpec(shape=(len(self.state.for_player(0)),), dtype=np.int32, minimum=0, name='observation'),
			array_spec.BoundedArraySpec(shape=(379,), dtype=np.int32, minimum=0, maximum=1, name='action_mask'))

	def run(self, step_limit=200):
		self.total_steps = 0
		self.start_game()
		while self.total_steps < step_limit:
			self.decide(self.immediate_play.pop(0) if self.immediate_play else self.current_player)
			if self.winning_player:
				self.end_game()
				self.start_game()
		self._reset()
		return self.total_steps

	def decide(self, active_player):
		self.num_step += 1
		self.total_steps += 1
		self.global_step += 1
		for player in self.player_list:
			player.start_trajectory()
		self.handler.handle_action(active_player.last_action, active_player)
		for player in self.player_list:
			player.end_trajectory()

	def build_phase(self, player_order):
		for build_player in player_order:
			build_player.dynamic_mask.only(df.place_settlement)
			np.logical_and(build_player.dynamic_mask.place_settlement, self.state.game_state_slices[df.vertex_open], out=build_player.dynamic_mask.place_settlement)
			self.decide(build_player)
			self.current_time_step_type = mid_step_type
			self.decide(build_player)
			build_player.dynamic_mask.only(df.no_action)

	def start_game(self):
		self._reset()
		print("starting game, current steps this run:", self.total_steps)
		self.episode_start = time.perf_counter()
		player_order = random.sample(self.player_list, len(self.player_list))
		self.player_cycle = cycle([x for x in player_order])

		self.build_phase(player_order)
		player_order.reverse()
		self.state.game_state_slices[df.build_phase_reversed].fill(1)
		self.build_phase(player_order)
		self.state.game_state_slices[df.build_phase].fill(0)
		self.state.game_state_slices[df.build_phase_reversed].fill(0)
		self.current_player = next(self.player_cycle)
		self.current_player.dynamic_mask.only(df.roll_dice)
		for player in self.player_list:
			player.static_mask.buy_development_card.fill(1)

	def end_game(self):
		end = time.perf_counter()
		for writer, player in zip(self.summary_list, self.player_list):
			writer(player.get_episode_summaries(), self.global_step)

		with self.writer.as_default():
			tf.summary.scalar(name="turn_count", data=self.state.game_state_slices[df.turn_number].item(), step=self.global_step.item())
			game_name = "game_{}".format(self.episode_number)
			# recap_data = [
			# 	[str(int(player.actual_victory_points)).ljust(2) for player in self.player_list],
			# 	[str(player.resource_cards).ljust(15) for player in self.player_list],
			# 	[str(player.development_cards_played.tolist()).ljust(15) for player in self.player_list],
			# 	[str(int(player.settlement_count)).ljust(3) for player in self.player_list],
			# 	[str(int(player.city_count)).ljust(3) for player in self.player_list],
			# 	[str(int(player.road_count)).ljust(3) for player in self.player_list],
			# 	[str(player.development_cards_played[knight_index])+"+" if player.owns_largest_army else " " for player in self.player_list],
			# 	[str(player.longest_road) + "+" if player.owns_longest_road else " " for player in self.player_list],
			# 	[str(int(player.policy_action_count / (player.implicit_action_count + 1e-9) * 1e2)) for player in self.player_list],
			# 	[player.episode_rewards for player in self.player_list],
			# 	[player.victory_points.item() for player in self.player_list],
			# 	[player.settlement_count.item() for player in self.player_list],
			# 	[player.city_count.item() for player in self.player_list],
			# 	[player.road_count.item() for player in self.player_list],
			# 	[np.sum(player.distribution_total).item() / self.state.game_state_slices[df.turn_number].item() for player in self.player_list],
			# 	[np.sum(player.steal_total).item() for player in self.player_list],
			# 	[np.sum(player.stolen_total).item() for player in self.player_list],
			# 	[np.sum(player.discard_total).item() / self.state.game_state_slices[df.turn_number].item() for player in self.player_list],
			# 	[np.sum(player.bank_trade_total).item() for player in self.player_list],
			# 	[np.sum(player.player_trade_total).item() for player in self.player_list],
			# 	[player.policy_action_count / player.implicit_action_count for player in self.player_list],
			# 	[player.longest_road / player.road_count.item() for player in self.player_list],
			# ]
			# recap_body = ""
			# recap_body += "|".join(["", *["Agent{}".format(player.index) for player in self.player_list]]) + "\n"
			# recap_body += "|:---|----:|----:|----:|" + "\n"
			# recap_body += "\n".join(["|".join([str(x) for x in data_line]) for data_line in recap_data])
			#
			# recap_body = """
			# 	### Markdown Text
			#
			# 	TensorBoard supports basic markdown syntax, including:
			#
			# 		preformatted code
			#
			# 	**bold text**
			#
			# 	| and | tables |
			# 	| ---- | ---------- |
			# 	| among | others |
			# 	"""
			#
			# recap_body = np.array([str(x) for x in range(16)])
			# recap_body = recap_body.reshape((4, 4))
			# recap_body = tf.convert_to_tensor(recap_body)

			# tf.summary.text(name=game_name, data=recap_body, step=self.total_steps)
			tf.summary.text(name=game_name, data="Steps/s: {}".format(int(self.num_step / (end - self.episode_start))), step=self.total_steps)
			tf.summary.text(name=game_name, data="Turns: {}".format(self.state.game_state_slices[df.turn_number].item()), step=self.total_steps)
			tf.summary.text(name=game_name, data="Steps: {}".format(str(self.num_step)), step=self.total_steps)
			tf.summary.text(name=game_name, data="Duration: {}s".format(str(end - self.episode_start)), step=self.total_steps)

		self.episode_number += 1

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
		self.development_card_stack = reverse_histogram(development_card_count_per_type)
		self.state.game_state_slices[df.bank_development_card_count] += sum(development_card_count_per_type)
		self.state.game_state_slices[df.bank_resources] += resource_card_count_per_type
		self.state.game_state_slices[df.build_phase].fill(1)
		self.num_step = 0
		self.state.game_state_slices[df.vertex_open].fill(1)
		self.state.game_state_slices[df.edge_open].fill(1)
		self.player_cycle = None
		self.current_player = None
		self.winning_player = None
		self.trading_player = None
		self.longest_road_owner = None
		self.largest_army_owner = None
		self.player_trades_this_turn = 0
		self.resolve_road_building_count = 0
		self.current_time_step_type = first_step_type
		self.immediate_play = []
		self.max_victory_points = 0
		self.episode_start = None

		for player in self.player_list: player.reset()
		for tile in self.board.tiles: tile.reset()
		for vertex in self.board.vertices: vertex.reset()
		for edge in self.board.edges: edge.reset()

	def get_observation(self, player):
		obs = np.expand_dims(self.state.for_player(player.index), axis=0)
		mask = np.expand_dims(player.dynamic_mask.mask, axis=0)
		obs = tf.convert_to_tensor(obs, dtype=tf.int32)
		mask = tf.convert_to_tensor(mask, dtype=tf.int32)
		return obs, mask

	def get_discount(self, exp_scale=2, offset=3):
		result = 1. - ((self.max_victory_points - offset) / self.victory_point_limit) ** exp_scale
		result = np.expand_dims(result, axis=0)
		result = tf.convert_to_tensor(result, dtype=tf.float32)
		return result

	def get_reward(self, player):
		reward = player.next_reward
		reward -= time_drain_reward
		reward /= 2
		player.episode_rewards += reward
		player.next_reward = 0
		return tf.convert_to_tensor(np.expand_dims(np.array(reward, dtype=np.double), axis=0), dtype=tf.float32)

	def get_time_step(self, player):
		return TimeStep(self.current_time_step_type, self.get_reward(player), self.get_discount(), self.get_observation(player))

