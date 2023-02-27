from pytan_fast import settings


def handle_place_road(edge, player_index, self):

	self.edge_locations[edge] = player_index

	if self.get_build_phase():
		self.build_phase_allowed_edges = None
		self.build_phase_road_placed = True
	if not self.get_build_phase():
		if not self.road_builder_block_state:
			self.trade_with_bank(player_index, settings.road_cost)

	self.road_builder_count += 1

	if self.road_builder_count == settings.road_building_road_count:
		self.road_builder_count = 0
		self.road_builder_responsible_index = -1
		self.road_builder_block_state = False

	self.set_edge_open(edge, 0)
	self.set_edge_owned(player_index, edge, 1)
	self.set_player_road_count(player_index, self.get_player_road_count(player_index) + 1)

	self.calculate_player_longest_road(player_index)
	# print(player_longest_road)
	# self.set_player_longest_road(player_index, player_longest_road)

	self.calculate_longest_road()
	self.calculate_victory_points()
