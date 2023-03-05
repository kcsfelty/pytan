import numpy as np
from tf_agents import trajectories
from tf_agents.trajectories import PolicyStep, StepType
import tensorflow as tf
import pytan_fast.definitions as df
from pytan_fast.mask import Mask
import pytan_fast.settings as gs
from pytan_fast.action_mapping import action_mapping

last_step_type = tf.convert_to_tensor(np.expand_dims(StepType.LAST, axis=0))

class Player:
	def __repr__(self):
		return "<Player{} (VP={} R={} D={} | S={} C={} R={} LA={}{} LR={}{} IAR={}%>".format(
			self.index,
			str(int(self.actual_victory_points)).ljust(2),
			str(self.resource_cards).ljust(15),
			str(self.development_cards_played.tolist()).ljust(15),
			str(int(self.settlement_count)).ljust(3),
			str(int(self.city_count)).ljust(3),
			str(int(self.road_count)).ljust(3),
			self.development_cards_played[gs.knight_index],
			"+" if self.owns_largest_army else " ",
			self.longest_road,
			"+" if self.owns_longest_road else " ",
			# np.sum(self.dynamic_mask.mask),
			str(int(self.policy_action_count / (self.implicit_action_count + 1e-9) * 1e2))
		)

	def __init__(self, index, policy, observers, game, public_state, private_state):
		self.index = index
		self.policy = policy
		self.observers = observers
		self.dynamic_mask = Mask()
		self.static_mask = Mask()
		self.game = game
		self.private_state = private_state
		self.public_state = public_state
		self.last_time_step = None
		self.last_action = None

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
		self.edge_list = []
		self.other_players = []
		self.actual_victory_points = 0
		self.episode_rewards = 0
		self.next_reward = 0
		self.current_action_is_implicit = False
		self.edge_proximity_vertices = []

		# Diagnostics / Statistics
		self.implicit_action_count = 0
		self.policy_action_count = 0

		# Summaries
		# Scalars
		self.distribution_total = np.zeros(gs.resource_type_count)
		self.steal_total = np.zeros(gs.resource_type_count)
		self.stolen_total = np.zeros(gs.resource_type_count)
		self.discard_total = np.zeros(gs.resource_type_count)
		self.bank_trade_total = np.zeros(gs.resource_type_count)
		self.player_trade_total = np.zeros(gs.resource_type_count)

		# Histograms
		self.action_count = []
		self.starting_distribution = np.zeros(gs.resource_type_count)

	def reset(self):
		self.edge_list = []
		self.implicit_action_count = 0
		self.policy_action_count = 0
		self.actual_victory_points = 0
		self.episode_rewards = 0
		self.next_reward = 0
		self.current_action_is_implicit = False
		self.action_count = []
		self.distribution_total = np.zeros(gs.resource_type_count)
		self.steal_total = np.zeros(gs.resource_type_count)
		self.stolen_total = np.zeros(gs.resource_type_count)
		self.discard_total = np.zeros(gs.resource_type_count)
		self.bank_trade_total = np.zeros(gs.resource_type_count)
		self.player_trade_total = np.zeros(gs.resource_type_count)
		self.starting_distribution = np.zeros(gs.resource_type_count)
		self.dynamic_mask = Mask()
		self.static_mask = Mask()

	def set_longest_road(self, has=True):
		if has:
			self.game.longest_road_owner = self
			self.change_victory_points(gs.longest_road_victory_points)
			self.owns_longest_road.fill(1)
		else:
			self.game.longest_road_owner = None
			self.change_victory_points(-1 * gs.longest_road_victory_points)
			self.owns_longest_road.fill(0)

	def largest_army(self, has=True):
		if has:
			self.game.largest_army_owner = self
			self.change_victory_points(gs.largest_army_victory_points)
			self.owns_largest_army.fill(1)
			np.copyto(self.game.state.game_state_slices[df.largest_army_size], self.development_cards_played[gs.knight_index])
		else:
			self.game.longest_road_owner = None
			self.change_victory_points(-1 * gs.largest_army_victory_points)
			self.owns_largest_army.fill(0)

	def change_victory_points(self, change):
		self.victory_points += change
		self.next_reward = change
		self.check_victory()

	def check_victory(self):
		victory_card_points = gs.victory_point_card_victory_points * self.development_cards[gs.victory_point_card_index]
		self.actual_victory_points = (victory_card_points + self.victory_points).item()
		if self.actual_victory_points > self.game.max_victory_points:
			self.game.max_victory_points = self.actual_victory_points
		if self.actual_victory_points >= gs.victory_points_to_win:
			self.development_cards_played[gs.victory_point_card_index] += self.development_cards[gs.victory_point_card_index]
			self.victory_points += victory_card_points
			self.next_reward += victory_card_points
			self.game.winning_player = self
			self.next_reward += 2 * (0.999 ** self.game.num_step) + 1
			self.game.current_time_step_type = last_step_type
			for opponent in self.other_players:
				opponent.next_reward -= 1

	def start_trajectory(self):
		self.last_time_step = self.game.get_time_step(self)
		if np.sum(self.dynamic_mask.mask) == 1:
			action_code = np.argmax(self.dynamic_mask.mask)
			action_code = tf.convert_to_tensor(action_code, dtype=tf.int32)
			action_code = tf.expand_dims(action_code, axis=0)
			self.last_action = PolicyStep(action=action_code)
			self.implicit_action_count += 1
			self.current_action_is_implicit = True
		else:
			self.current_action_is_implicit = False
			self.last_action = self.policy.action(self.last_time_step)
			self.policy_action_count += 1

	def end_trajectory(self):
		if not self.current_action_is_implicit:
			next_time_step = self.game.get_time_step(self)
			traj = trajectories.from_transition(self.last_time_step, self.last_action, next_time_step)
			for observer in self.observers:
				observer(traj)

	def can_afford(self, trade):
		return np.all(self.resource_cards + trade >= 0)

	def apply_static(self, term):
		np.logical_and(self.dynamic_mask.mask_slices[term], self.static_mask.mask_slices[term], out=self.dynamic_mask.mask_slices[term])

	def calculate_longest_road(self):
		player_vertex_list = []
		for edge in self.edge_list:
			for touch_vertex in edge.vertices:
				if touch_vertex not in player_vertex_list:
					player_vertex_list.append(touch_vertex)
		paths = []
		for start_vertex in self.edge_proximity_vertices:
			paths_from_this_node = []
			agenda = [(start_vertex, [])]
			while len(agenda) > 0:
				vertex, path_thus_far = agenda.pop()
				able_to_navigate = False
				for neighbor_vertex in vertex.vertices:
					vertices_edge = vertex.edge_between_vertex[neighbor_vertex]
					if vertices_edge not in self.edge_list:
						continue
					if vertices_edge not in path_thus_far:
						agenda.insert(0, (neighbor_vertex, path_thus_far + [vertices_edge]))
						able_to_navigate = True
				if not able_to_navigate:
					paths_from_this_node.append(path_thus_far)
			paths.extend(paths_from_this_node)
		self.longest_road = len(max(paths, key=len))

	def get_episode_summaries(self):
		scalars = {
			"episode_rewards": self.episode_rewards,
			"victory_points": self.victory_points.item(),
			"settlements": self.settlement_count.item(),
			"cities": self.city_count.item(),
			"roads": self.road_count.item(),
			"distribution_per_turn_total": np.sum(self.distribution_total).item() / self.game.state.game_state_slices[df.turn_number].item(),
			"steal_total": np.sum(self.steal_total).item(),
			"stolen_total": np.sum(self.stolen_total).item(),
			"discard_total": np.sum(self.discard_total).item() / self.game.state.game_state_slices[df.turn_number].item(),
			"bank_trade_total": np.sum(self.bank_trade_total).item(),
			"player_trade_total": np.sum(self.player_trade_total).item(),
			"implicit_action_ratio": self.policy_action_count / self.implicit_action_count,
			"longest_road": self.longest_road,
		}
		histograms = {
			"action_count": self.action_count,
		}
		return {
			"scalars": scalars,
			"histograms": histograms
		}

