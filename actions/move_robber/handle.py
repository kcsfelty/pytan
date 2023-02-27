from pytan_fast import settings
from geometry.get_tile_vertices import get_tile_vertices


def handle_move_robber(tile, player_index, self):
	self.move_robber_block_state = False
	self.rob_player_block_state = True
	self.rob_player_responsible = player_index
	self.set_tile_has_robber(self.robber_position, 0)
	self.robber_position = tile
	self.set_tile_has_robber(tile, 1)
	self.set_player_must_move_robber(player_index, 0)
	self.players_to_rob = []
	for vertex in get_tile_vertices(*tile):
		if vertex in self.vertex_list:
			if not self.get_vertex_open(vertex):
				for owner_index in settings.player_list:
					if not owner_index == player_index:
						if self.get_vertex_owned(owner_index, vertex):
							if owner_index not in self.players_to_rob:
								self.players_to_rob.append(owner_index)

	if len(self.players_to_rob) == 0:
		self.rob_player_block_state = False
		self.players_to_rob = []
		self.rob_player_responsible = -1
	if len(self.players_to_rob) == 1:
		if self.get_player_resource_card_count(self.players_to_rob[0]) > 0:
			self.rob_player(player_index, self.players_to_rob[0])
		self.rob_player_block_state = False
		self.players_to_rob = []
		self.rob_player_responsible = -1
