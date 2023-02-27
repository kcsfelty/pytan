import numpy as np
from tf_agents import trajectories
from tf_agents.trajectories import PolicyStep
import tensorflow as tf
import pytan_fast.definitions as df
from pytan_fast.mask import Mask


class Player:
	def __repr__(self):
		return "<Player_{}: (VP={} RCC={} DCC={} S={} C={} R={} LA={} LR={} IAR={}%)>".format(
			self.index,
			int(self.actual_victory_points),
			self.resource_cards.tolist(),
			self.development_cards_played.tolist(),
			int(self.settlement_count),
			int(self.city_count),
			int(self.road_count),
			"+" if self.owns_largest_army else " ",
			"+" if self.owns_longest_road else " ",
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
		self.vertex_list = []
		self.edge_list = []
		self.other_players = []
		self.actual_victory_points = 0
		self.episode_rewards = 0

		# Diagnostics
		self.implicit_action_count = 0
		self.policy_action_count = 0

	def reset(self):
		self.vertex_list = []
		self.edge_list = []
		self.implicit_action_count = 0
		self.policy_action_count = 0
		self.actual_victory_points = 0
		self.episode_rewards = 0

	def start_trajectory(self):
		self.last_time_step = self.game.get_time_step(self)
		if np.sum(self.dynamic_mask.mask) == 1:
			action_code = np.argmax(self.dynamic_mask.mask)
			action_code = tf.convert_to_tensor(action_code, dtype=tf.int32)
			action_code = tf.expand_dims(action_code, axis=0)
			self.last_action = PolicyStep(action=action_code)
			self.implicit_action_count += 1
		else:
			self.last_action = self.policy.action(self.last_time_step)
			self.policy_action_count += 1

	def end_trajectory(self):
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
		for start_vertex in player_vertex_list:
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

