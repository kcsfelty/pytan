import numpy as np
import pytan_fast.settings as gs
import pytan_fast.definitions as df
from pytan_fast.states.game_state import game_state_degrees, game_state_degrees_condensed
from pytan_fast.states.private_state import private_state_degrees
from pytan_fast.states.public_state import public_state_degrees, public_state_degrees_condensed


class State:
	def __init__(self, game_count=1, player_count=3, condensed_state=False):
		self.game_count = game_count
		self.condensed_state = condensed_state

		if self.condensed_state:
			standard_game_state_degrees = game_state_degrees_condensed
			extended_game_state_degrees = game_state_degrees
			standard_public_state_degrees = public_state_degrees_condensed
			extended_public_state_degrees = public_state_degrees
		else:
			standard_game_state_degrees = game_state_degrees
			extended_game_state_degrees = game_state_degrees_condensed
			standard_public_state_degrees = public_state_degrees
			extended_public_state_degrees = public_state_degrees_condensed

		# combined_game_state_degrees = standard_game_state_degrees | extended_game_state_degrees
		# extra_game_state_degrees = extended_game_state_degrees.keys() - standard_game_state_degrees.keys()
		combined_public_state_degrees = standard_public_state_degrees | extended_public_state_degrees
		extra_public_state_degrees = extended_public_state_degrees.keys() - standard_public_state_degrees.keys()

		# print(standard_game_state_degrees)
		# print(standard_public_state_degrees)
		self.observation_terms = []
		for term in standard_game_state_degrees:
			if standard_game_state_degrees[term] > 1:
				self.observation_terms.extend([term + "_" + str(i) for i in range(standard_game_state_degrees[term])])
			else:
				self.observation_terms.append(term)
		for term in private_state_degrees:
			if private_state_degrees[term] > 1:
				self.observation_terms.extend([term + "_" + str(i) for i in range(private_state_degrees[term])])
			else:
				self.observation_terms.append(term)
		for term in standard_public_state_degrees:
			if standard_public_state_degrees[term] > 1:
				self.observation_terms.extend([term + "_" + str(i) for i in range(standard_public_state_degrees[term])])
			else:
				self.observation_terms.append(term)
		# for term in self.observation_terms:
			# print(term)
		# self.extra_game_state = {}
		# for game_state_term in extra_game_state_degrees:
		# 	self.extra_game_state[game_state_term] = np.zeros(combined_game_state_degrees[game_state_term])
			# print(game_state_term)
		# print()
		self.extra_public_state = {}
		for public_state_term in extra_public_state_degrees:
			self.extra_public_state[public_state_term] = np.zeros((gs.player_count, combined_public_state_degrees[public_state_term]), dtype=np.int32)

		self.game_state_len = sum([standard_game_state_degrees[term] for term in standard_game_state_degrees])
		self.private_state_len = sum([private_state_degrees[term] for term in private_state_degrees])
		self.public_state_len = sum([standard_public_state_degrees[term] for term in standard_public_state_degrees])
		total_len = self.game_state_len + gs.player_count * (self.private_state_len + self.public_state_len)
		self.state = np.zeros((game_count, total_len), dtype=np.int32)
		self.state_slices = {}
		self.state_slices[df.game_state] = self.state[:, :self.game_state_len]
		self.state_slices[df.private_state] = self.state[:, self.game_state_len:self.game_state_len + self.private_state_len * gs.player_count]
		self.state_slices[df.public_state] = self.state[:, self.game_state_len + self.private_state_len * gs.player_count:]
		self.state_slices[df.game_state].shape = (game_count, self.game_state_len)
		self.state_slices[df.private_state].shape = (game_count, gs.player_count, self.private_state_len)
		self.state_slices[df.public_state].shape = (game_count, gs.player_count, self.public_state_len)
		self.game_state_slices = {}
		current_index = 0
		for term in standard_game_state_degrees:
			next_index = current_index + standard_game_state_degrees[term]
			self.game_state_slices[term] = self.state_slices[df.game_state][:, current_index:next_index]
			current_index = next_index

		self.game_state_slices[df.tile_resource].shape = (game_count, gs.tile_count, gs.resource_type_count_tile)
		self.game_state_slices[df.tile_roll_number].shape = (game_count, gs.tile_count, gs.roll_number_count)
		self.game_state_slices[df.vertex_has_port].shape = (game_count, gs.vertex_count, gs.port_type_count)

		self.bank_development_card_count = self.game_state_slices[df.bank_development_card_count]
		self.longest_road_length = self.game_state_slices[df.longest_road_length]
		self.largest_army_size = self.game_state_slices[df.largest_army_size]
		self.turn_number = self.game_state_slices[df.turn_number]
		self.build_phase = self.game_state_slices[df.build_phase]
		self.build_phase_reversed = self.game_state_slices[df.build_phase_reversed]
		self.build_phase_placed_settlement = self.game_state_slices[df.build_phase_placed_settlement]
		self.build_phase_placed_road = self.game_state_slices[df.build_phase_placed_road]
		self.bought_development_card_count = self.game_state_slices[df.bought_development_card_count]
		self.played_development_card_count = self.game_state_slices[df.played_development_card_count]
		self.vertex_settlement = self.game_state_slices[df.vertex_settlement]
		self.vertex_city = self.game_state_slices[df.vertex_city]
		self.vertex_open = self.game_state_slices[df.vertex_open]
		self.edge_open = self.game_state_slices[df.edge_open]
		self.tile_has_robber = self.game_state_slices[df.tile_has_robber]
		self.tile_resource = self.game_state_slices[df.tile_resource]
		self.tile_roll_number = self.game_state_slices[df.tile_roll_number]
		self.bank_resources = self.game_state_slices[df.bank_resources]
		self.current_player_trade = self.game_state_slices[df.current_player_trade]
		self.current_roll = self.game_state_slices[df.current_roll]
		self.vertex_has_port = self.game_state_slices[df.vertex_has_port]

		self.private_state_slices = []
		for player_index in gs.player_list:
			private_state_slice = {}
			current_index = 0
			for term in private_state_degrees:
				next_index = current_index + private_state_degrees[term]
				private_state_slice[term] = self.state_slices[df.private_state][:, player_index, current_index:next_index]
				current_index = next_index
			self.private_state_slices.append(private_state_slice)
		self.public_state_slices = []
		for player_index in gs.player_list:
			current_index = 0
			public_state_slice = {}
			for term in standard_public_state_degrees:
				next_index = current_index + standard_public_state_degrees[term]
				public_state_slice[term] = self.state_slices[df.public_state][:, player_index, current_index:next_index]
				current_index = next_index

			self.public_state_slices.append(public_state_slice)
		# self.observation_array = np.array(self.state_slices[df.public_state], dtype=np.int32)

	def for_player(self, player_index):
		observation_order = [i for i in gs.player_list]
		observation_order.insert(player_index, observation_order.pop(observation_order.index(player_index)))
		game_view = self.state_slices[df.game_state].copy()
		private_view = self.state_slices[df.private_state][:, player_index].copy()
		public_view = self.state_slices[df.public_state][:, observation_order].copy()
		public_view.shape = (self.game_count, self.public_state_len * gs.player_count)
		return np.hstack([game_view, private_view, public_view])

	def reset(self, game_index):
		self.state[game_index].fill(0)
