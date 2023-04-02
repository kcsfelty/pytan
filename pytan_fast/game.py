import random
import random
import time
from abc import ABC
from itertools import cycle

import numpy as np
import tensorflow as tf
from tf_agents.environments import PyEnvironment
from tf_agents.specs import BoundedArraySpec
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


def reverse_histogram(hist):
	hist = np.repeat([x for x in range(len(hist))], hist).tolist()
	rng.shuffle(hist)
	return hist


first_step_type = tf.convert_to_tensor(np.expand_dims(StepType.FIRST, axis=0))
mid_step_type = tf.convert_to_tensor(np.expand_dims(StepType.MID, axis=0))
last_step_type = tf.convert_to_tensor(np.expand_dims(StepType.LAST, axis=0))


class PyTanFast(PyEnvironment, ABC):
	def __init__(self, agent_list=None, global_step=None, log_dir="./logs", victory_point_limit=10, condensed_state=False, env_index=None, eval=False):
		super().__init__()

		# Summaries
		self.log_dir = log_dir + "/game"
		self.episode_number = 0
		self.writer = tf.compat.v2.summary.create_file_writer(self.log_dir)

		# Environment
		self.env_index = env_index
		self.eval = eval
		self.global_step = global_step
		self.condensed_state = condensed_state
		self.state = State(self.condensed_state)
		self.board = Board(self.state, self)
		self.agent_list = agent_list or [None] * 3
		self.player_list = None
		self.victory_point_limit = victory_point_limit

		if not self.player_list:
			self.player_list = [Player(
				index=player_index,
				game=self,
				agent=self.agent_list[player_index],
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

		# Game logic helpers
		self.development_card_stack = []
		self.trading_player = None
		self.longest_road_owner = None
		self.largest_army_owner = None
		self.player_trades_this_turn = 0
		self.resolve_road_building_count = 0
		self.max_victory_points = 0
		self.bank_resource_tuple = tuple(self.state.bank_resources)

		self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=379 - 1, name='action'),

		self._observation_spec = (
			BoundedArraySpec(shape=(len(self.state.for_player(0)),), dtype=np.int32, minimum=0, maximum=128, name='observation'),
			BoundedArraySpec(shape=(379,), dtype=np.int32, minimum=0, maximum=1, name='action_mask'))

	def walk(self):
		self.decide(self.immediate_play.pop(0) if self.immediate_play else self.current_player)
		if self.winning_player:
			self.end_game()
			self._reset()

	def run(self, step_limit=600):
		self.total_steps = 0
		while self.total_steps < step_limit:
			self.walk()
		return self.total_steps

	def decide(self, active_player):
		self.num_step += 1
		self.total_steps += 1
		self.global_step.assign_add(1)
		for player in self.player_list:
			player.start_trajectory()
		self.handler.handle_action(active_player.last_action, active_player)
		for player in self.player_list:
			player.end_trajectory()

	def build_phase(self, player_order):
		for build_player in player_order:
			build_player.dynamic_mask.only(df.place_settlement)
			np.logical_and(build_player.dynamic_mask.place_settlement, self.state.vertex_open, out=build_player.dynamic_mask.place_settlement)
			self.decide(build_player)
			self.current_time_step_type = mid_step_type
			self.decide(build_player)
			build_player.dynamic_mask.only(df.no_action)

	def start_game(self):
		player_order = random.sample(self.player_list, len(self.player_list))
		self.player_cycle = cycle([x for x in player_order])

		self.build_phase(player_order)
		player_order.reverse()
		self.state.build_phase_reversed.fill(1)
		self.build_phase(player_order)
		self.state.build_phase.fill(0)
		self.state.build_phase_reversed.fill(0)
		self.current_player = next(self.player_cycle)
		self.current_player.dynamic_mask.only(df.roll_dice)
		for player in self.player_list:
			player.static_mask.buy_development_card.fill(1)

	def end_game(self):
		for player in self.player_list:
			player.agent.write_summary(player.get_episode_summaries(), self.global_step)
		self.write_episode_summary()
		self.episode_number += 1

	def write_episode_summary(self):
		summary = ""
		if self.env_index is not None:
			summary += "[env_{}]".format(self.env_index)
		summary += "Game finished"
		summary += str(self.state.turn_number.item()).rjust(5)
		summary += " turns, "
		summary += str(self.num_step).rjust(6)
		summary += ", "
		summary += str(self.winning_player)
		print(summary)

		with self.writer.as_default():
			tf.summary.scalar(
				name="turn_count",
				data=self.state.turn_number.item(),
				step=self.global_step.numpy().item())

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
		self.state.bank_development_card_count += sum(development_card_count_per_type)
		self.state.bank_resources += resource_card_count_per_type
		self.state.build_phase.fill(1)
		self.num_step = 0
		self.state.vertex_open.fill(1)
		self.state.edge_open.fill(1)
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
		self.bank_resource_tuple = tuple(self.state.bank_resources)

		for player in self.player_list: player.reset()
		for tile in self.board.tiles: tile.reset()
		for vertex in self.board.vertices: vertex.reset()
		for edge in self.board.edges: edge.reset()

		self.start_game()

	def get_observation(self, player):
		obs = np.expand_dims(self.state.for_player(player.index), axis=0)
		mask = np.expand_dims(player.dynamic_mask.mask, axis=0)
		obs = tf.convert_to_tensor(obs, dtype=tf.int32)
		mask = tf.convert_to_tensor(mask, dtype=tf.int32)
		return obs, mask

	def get_discount(self, exp_scale=2, offset=6):
		# result = 1. - ((self.max_victory_points - offset) / self.victory_point_limit) ** exp_scale
		result = 1.0
		result = np.expand_dims(result, axis=0)
		result = tf.convert_to_tensor(result, dtype=tf.float32)
		return result

	def get_reward(self, player):
		reward = player.next_reward
		reward -= time_drain_reward
		reward *= 1
		player.episode_rewards += reward
		player.next_reward = 0
		return tf.convert_to_tensor(np.expand_dims(np.array(reward, dtype=np.double), axis=0), dtype=tf.float32)

	def get_time_step(self, player):
		return TimeStep(self.current_time_step_type, self.get_reward(player), self.get_discount(), self.get_observation(player))

	def get_player_win_rate_order(self, n=50):
		win_rates = [{"index": player.index, "win_rate": player.win_rate(n)} for player in self.player_list]
		return [player_win_dict["index"] for player_win_dict in sorted(win_rates, key=lambda x: x["win_rate"], reverse=True)]
