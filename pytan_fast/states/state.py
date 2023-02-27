import numpy as np
import pytan_fast.settings as gs
import pytan_fast.definitions as df
from pytan_fast.states.game_state import game_state_degrees
from pytan_fast.states.private_state import private_state_degrees
from pytan_fast.states.public_state import public_state_degrees


class State:
	def __init__(self):
		self.game_state_len = sum([game_state_degrees[term] for term in game_state_degrees])
		self.private_state_len = sum([private_state_degrees[term] for term in private_state_degrees])
		self.public_state_len = sum([public_state_degrees[term] for term in public_state_degrees])
		total_len = self.game_state_len + gs.player_count * (self.private_state_len + self.public_state_len)
		self.state = np.zeros(total_len, dtype=np.int32)
		self.state_slices = {}
		self.state_slices[df.game_state] = self.state[:self.game_state_len]
		self.state_slices[df.private_state] = self.state[self.game_state_len:self.game_state_len + self.private_state_len * gs.player_count]
		self.state_slices[df.public_state] = self.state[self.game_state_len + self.private_state_len * gs.player_count:]
		self.state_slices[df.private_state].shape = (gs.player_count, self.private_state_len)
		self.state_slices[df.public_state].shape = (gs.player_count, self.public_state_len)
		self.game_state_slices = {}
		current_index = 0
		for term in game_state_degrees:
			next_index = current_index + game_state_degrees[term]
			self.game_state_slices[term] = self.state_slices[df.game_state][current_index:next_index]
			current_index = next_index

		self.game_state_slices[df.tile_resource].shape = (gs.tile_count, gs.resource_type_count_tile)
		self.game_state_slices[df.tile_roll_number].shape = (gs.tile_count, gs.roll_number_count)
		self.game_state_slices[df.vertex_has_port].shape = (gs.vertex_count, gs.port_type_count)

		self.private_state_slices = []
		for player_index in gs.player_list:
			private_state_slice = {}
			current_index = 0
			for term in private_state_degrees:
				next_index = current_index + private_state_degrees[term]
				private_state_slice[term] = self.state_slices[df.private_state][player_index, current_index:next_index]
				current_index = next_index
			self.private_state_slices.append(private_state_slice)
		self.public_state_slices = []
		for player_index in gs.player_list:
			current_index = 0
			public_state_slice = {}
			for term in public_state_degrees:
				next_index = current_index + public_state_degrees[term]
				public_state_slice[term] = self.state_slices[df.public_state][player_index, current_index:next_index]
				current_index = next_index
			self.public_state_slices.append(public_state_slice)

	def for_player(self, player_index):
		observation_order = [i for i in gs.player_list]
		observation_order.insert(player_index, observation_order.pop(observation_order.index(player_index)))
		game_view = self.state_slices[df.game_state].copy()
		private_view = self.state_slices[df.private_state][player_index].copy()
		public_view = self.state_slices[df.public_state][observation_order].copy()
		public_view.shape = (self.public_state_len * gs.player_count)
		return np.hstack([game_view, private_view, public_view])

	def reset(self):
		self.state.fill(0)
