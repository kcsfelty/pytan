import numpy as np

import reference.definitions as df
import reference.settings as gs
from game.mask import Mask
import tensorflow as tf

class Player:
	def __repr__(self):
		return "<Player{}>".format(self.index)

	def for_game(self, game_index):
		return "{}Player{} S={} C={} R={} VC={} LA={}{} LR={}{}".format(
			"@" if self.current_player[game_index] else " ",
			self.index,
			str(int(self.settlement_count[game_index])).ljust(3),
			str(int(self.city_count[game_index])).ljust(3),
			str(int(self.road_count[game_index])).ljust(3),
			str(self.development_cards_played[game_index][gs.victory_point_card_index].item()).ljust(2),
			str(self.development_cards_played[game_index][gs.knight_index]).rjust(2),
			"*" if self.owns_largest_army[game_index] else " ",
			str(self.longest_road[game_index].item()).rjust(2),
			"*" if self.owns_longest_road[game_index] else " ")

	def __str__(self):
		return self.__repr__()

	def __init__(self, index, game, public_state, private_state):
		self.index = index
		self.game = game
		self.dynamic_mask = Mask(self.game.game_count)
		self.static_mask = Mask(self.game.game_count)
		self.private_state = private_state
		self.public_state = public_state

		self.current_player = self.public_state[df.current_player]
		self.must_move_robber = self.public_state[df.must_move_robber]
		self.victory_points = self.public_state[df.victory_points]
		self.resource_card_count = self.public_state[df.resource_card_count]
		self.development_card_count = self.public_state[df.development_card_count]
		self.settlement_count = self.public_state[df.settlement_count]
		self.city_count = self.public_state[df.city_count]
		self.road_count = self.public_state[df.road_count]
		self.longest_road = self.public_state[df.longest_road]
		self.owns_longest_road = self.public_state[df.owns_longest_road]
		self.owns_largest_army = self.public_state[df.owns_largest_army]
		self.offering_trade = self.public_state[df.offering_trade]
		self.accepted_trade = self.public_state[df.accepted_trade]
		self.declined_trade = self.public_state[df.declined_trade]
		self.must_discard = self.public_state[df.must_discard]
		self.development_cards_played = self.public_state[df.development_cards_played]
		self.port_access = self.public_state[df.port_access]
		self.owned_vertices = self.public_state[df.vertex_owned]
		self.owned_edges = self.public_state[df.edge_owned]
		self.resource_cards = self.private_state[df.resource_type_count]
		self.development_cards = self.private_state[df.development_type_count]
		self.development_card_bought_this_turn = self.private_state[df.development_type_bought_count]

		# Helper fields
		self.other_players = []
		self.win_list = []
		self.edge_list = [[] for _ in range(self.game.game_count)]
		self.actual_victory_points = [0 for _ in range(self.game.game_count)]
		self.edge_proximity_vertices = [[] for _ in range(self.game.game_count)]

		# Diagnostics / Statistics
		self.implicit_action_count = np.zeros((self.game.game_count,))
		self.policy_action_count = np.zeros((self.game.game_count,))
		self.writer = tf.summary.create_file_writer(logdir=self.game.log_dir + "/player{}".format(str(self.index)))

		# Summaries
		# Scalars
		self.distribution_total = np.zeros((self.game.game_count, gs.resource_type_count))
		self.steal_total = np.zeros((self.game.game_count, gs.resource_type_count))
		self.stolen_total = np.zeros((self.game.game_count, gs.resource_type_count))
		self.discard_total = np.zeros((self.game.game_count, gs.resource_type_count))
		self.bank_trade_total = np.zeros((self.game.game_count, gs.resource_type_count))
		self.player_trade_total = np.zeros((self.game.game_count, gs.resource_type_count))

		# Histograms
		self.starting_distribution = np.zeros((self.game.game_count, gs.resource_type_count))

	def reset(self, game_index):
		self.policy_action_count = 0
		self.implicit_action_count = 0
		self.edge_list[game_index] = []
		self.actual_victory_points[game_index] = 0
		self.distribution_total[game_index].fill(0)
		self.steal_total[game_index].fill(0)
		self.stolen_total[game_index].fill(0)
		self.discard_total[game_index].fill(0)
		self.bank_trade_total[game_index].fill(0)
		self.player_trade_total[game_index].fill(0)
		self.starting_distribution[game_index].fill(0)
		self.dynamic_mask.reset(game_index)
		self.static_mask.reset(game_index)

	def set_longest_road(self, has, game_index):
		if has:
			self.game.longest_road_owner[game_index] = self
			self.change_victory_points(gs.longest_road_victory_points, game_index)
			self.owns_longest_road[game_index].fill(1)
		else:
			self.game.longest_road_owner[game_index] = None
			self.change_victory_points(-1 * gs.longest_road_victory_points, game_index)
			self.owns_longest_road[game_index].fill(0)

	def largest_army(self, has, game_index):
		if has:
			self.game.largest_army_owner[game_index] = self
			self.change_victory_points(gs.largest_army_victory_points, game_index)
			self.owns_largest_army[game_index].fill(1)
			np.copyto(self.game.state.largest_army_size[game_index], self.development_cards_played[game_index][gs.knight_index])
		else:
			self.game.longest_road_owner[game_index] = None
			self.change_victory_points(-1 * gs.largest_army_victory_points, game_index)
			self.owns_largest_army[game_index].fill(0)

	def change_victory_points(self, change, game_index):
		self.victory_points[game_index] += change
		# self.game.reward[self.index][game_index] += float(change)
		self.check_victory(game_index)

	def check_victory(self, game_index):
		victory_card_points = gs.victory_point_card_victory_points * self.development_cards[game_index][gs.victory_point_card_index]
		self.actual_victory_points[game_index] = (victory_card_points + self.victory_points[game_index]).item()
		if self.actual_victory_points[game_index] >= gs.victory_points_to_win:
			self.development_cards_played[game_index][gs.victory_point_card_index] += self.development_cards[game_index][gs.victory_point_card_index]
			self.victory_points[game_index] += victory_card_points
			# self.game.reward[self.index][game_index] += float(victory_card_points)
			self.game.winning_player[game_index] = self

	def can_afford(self, trade, game_index):
		return np.all(self.resource_cards[game_index] + trade >= 0)

	def apply_static(self, term, game_index):
		np.logical_and(
			self.dynamic_mask.mask_slices[term][game_index],
			self.static_mask.mask_slices[term][game_index],
			out=self.dynamic_mask.mask_slices[term][game_index])

	def calculate_longest_road(self, game_index):
		paths = []
		for start_vertex in self.edge_proximity_vertices[game_index]:
			paths_from_this_node = []
			agenda = [(start_vertex, [])]
			while len(agenda) > 0:
				vertex, path_thus_far = agenda.pop()
				able_to_navigate = False
				for neighbor_vertex in vertex.vertices:
					vertices_edge = vertex.edge_between_vertex[neighbor_vertex]
					if not self.owned_edges[game_index][vertices_edge.index]:
						continue
					if neighbor_vertex.owned_by[game_index] in self.other_players:
						continue
					if vertices_edge not in path_thus_far:
						agenda.insert(0, (neighbor_vertex, path_thus_far + [vertices_edge]))
						able_to_navigate = True
				if not able_to_navigate:
					paths_from_this_node.append(path_thus_far)
			paths.extend(paths_from_this_node)
		self.longest_road[game_index] = len(max(paths, key=len))

	def write_episode_summary(self, game_index):
		if self.game.winning_player[game_index] == self:
			self.win_list.append(1)
		scalars = {
			"victory_points": self.victory_points[game_index].item(),
			"settlements": self.settlement_count[game_index].item(),
			"cities": self.city_count[game_index].item(),
			"roads": self.road_count[game_index].item(),
			"longest_road": self.longest_road[game_index].item(),
			"owns_longest_road": self.owns_longest_road[game_index].item(),
			"owns_largest_army": self.owns_largest_army[game_index].item(),
			"knights_played": self.development_cards_played[game_index][gs.knight_index].item(),
			"monopoly_played": self.development_cards_played[game_index][gs.monopoly_index].item(),
			"year_of_plenty_played": self.development_cards_played[game_index][gs.year_of_plenty_index].item(),
			"road_building_played": self.development_cards_played[game_index][gs.road_building_index].item(),
			"victory_cards_played": self.development_cards_played[game_index][gs.victory_point_card_index].item(),
			# "distribution_avg": np.sum(self.distribution_total).item() / self.game.state.turn_number.item(),
			# "steal_avg": np.sum(self.steal_total).item() / self.game.state.turn_number.item(),
			# "stolen_avg": np.sum(self.stolen_total).item() / self.game.state.turn_number.item(),
			# "discard_avg": np.sum(self.discard_total).item() / self.game.state.turn_number.item(),
			# "bank_trade_total": np.sum(self.bank_trade_total).item(),
			# "player_trade_total": np.sum(self.player_trade_total).item(),
			# "implicit_action_ratio": self.policy_action_count / self.implicit_action_count,
			# "longest_road_per_road": self.longest_road[game_index] / self.road_count[game_index].item(),
			"win_rate_50": self.win_rate(50),
		}

		if self.game.winning_player[game_index] == self:
			self.win_list.append(1)
			scalars["turn_count"] = self.game.state.turn_number[game_index].item()

		step = tf.cast(self.game.global_step, dtype=tf.int64)

		with self.writer.as_default(step):
			for key in scalars:
				tf.summary.scalar(name=key, data=float(scalars[key]))

	def win_rate(self, n):
		return self.avg_last_n(self.win_list, n)

	def avg_last_n(self, values, n):
		last_values = values[-n:]
		if len(last_values) == 0:
			return 0
		return sum(last_values) / len(last_values)
