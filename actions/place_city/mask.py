from game import game_settings
from game.game_settings import city_cost


def mask_place_city(vertex):
	from game.pytan.__init__ import PyTan

	def callback_place_city(self: PyTan, player_index):
		if not self.get_player_is_current_player(player_index):
			return False
		if not self.player_can_afford(player_index, city_cost):
			return False
		if self.get_vertex_city(vertex):
			return False
		if not self.get_vertex_settlement(vertex):
			return False
		if not self.get_player_city_count(player_index) <= game_settings.max_city_count:
			return False
		if not self.get_vertex_owned(player_index, vertex):
			return False
		return True

	return callback_place_city
