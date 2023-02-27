from game import game_settings
from game.game_settings import settlement_cost


def mask_place_settlement(vertex):
	from game.pytan.__init__ import PyTan

	def callback_place_settlement(self: PyTan, player_index):
		if not self.get_vertex_open(vertex):
			return False
		if not self.get_player_is_current_player(player_index):
			return False
		if self.get_build_phase():
			if self.build_phase_allowed_edges:
				return False
			if self.build_phase_settlement_placed:
				return False
			return True
		if not self.get_player_settlement_count(player_index) < game_settings.max_settlement_count:
			return False
		if not self.player_can_afford(player_index, settlement_cost):
			return False
		for adjacent_edge in self.vertex_edges[vertex]:
			if self.get_edge_owned(player_index, adjacent_edge):
				return True
		return False

	return callback_place_settlement
