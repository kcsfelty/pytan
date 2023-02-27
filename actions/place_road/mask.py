from game import game_settings
from game.game_settings import road_cost
from geometry.get_edge_vertices import get_edge_vertices


def mask_place_road(edge):
	from game.pytan.__init__ import PyTan

	def callback_place_road(self: PyTan, player_index):
		if not self.get_player_road_count(player_index) < game_settings.max_road_count:
			return False
		if edge not in self.edge_list:
			return False
		if not self.get_edge_open(edge):
			return False
		if not self.get_player_is_current_player(player_index):
			return False

		if self.get_build_phase():
			if not self.build_phase_settlement_placed:
				return False
			if not self.build_phase_allowed_edges:
				return False
			if edge in self.build_phase_allowed_edges:
				return True
			return False
		if self.road_builder_block_state:
			if self.road_builder_count < game_settings.road_building_road_count:
				return False
		else:
			if not self.player_can_afford(player_index, road_cost):
				return False
		for adjacent_edge in self.edge_edges[edge]:
			if self.get_edge_owned(player_index, adjacent_edge):
				edges_vertex = get_edges_vertex(edge, adjacent_edge)
				if not self.get_vertex_open(edges_vertex):
					for opponent_index in game_settings.player_list:
						if opponent_index is not player_index:
							if self.get_vertex_owned(opponent_index, edges_vertex):
								return False
				return True
		return False

	return callback_place_road


def get_edges_vertex(edge1, edge2):
	edge1_vertices = get_edge_vertices(*edge1)
	edge2_vertices = get_edge_vertices(*edge2)
	for e1v in edge1_vertices:
		for e2v in edge2_vertices:
			if e1v == e2v:
				return e1v
