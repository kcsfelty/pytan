from pytan_fast import settings


def handle_place_city(vertex, player_index, self):
	self.city_locations[vertex] = player_index
	self.trade_with_bank(player_index, settings.city_cost)
	self.set_vertex_settlement(vertex, 0)
	self.set_vertex_city(vertex, 1)
	new_count = self.get_player_city_count(player_index) + 1
	self.set_player_city_count(player_index, new_count)
	old_count = self.get_player_settlement_count(player_index) - 1
	self.set_player_settlement_count(player_index, old_count)
	self.calculate_victory_points()

