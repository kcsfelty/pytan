from game import game_settings


def mask_play_road_building():
	from game.pytan.__init__ import PyTan

	def callback_play_road_building(self: PyTan, player_index):
		if self.get_player_road_count(player_index) >= game_settings.max_road_count - 1:
			return False
		if not self.player_rolled:
			return False
		if not self.get_player_is_current_player(player_index):
			return False
		if not self.get_private_development_card_count(player_index, game_settings.road_building_index) > 0:
			return False
		return True

	return callback_play_road_building
