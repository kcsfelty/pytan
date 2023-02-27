from game import game_settings


def mask_play_monopoly(resource_index):
	from game.pytan.__init__ import PyTan

	def callback_play_monopoly(self: PyTan, player_index):
		if not self.player_rolled:
			return False
		if not self.get_player_is_current_player(player_index):
			return False
		if not self.get_private_development_card_count(player_index, game_settings.monopoly_index) > 0:
			return False
		return True

	return callback_play_monopoly
