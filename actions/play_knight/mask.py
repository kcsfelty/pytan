from game import game_settings


def mask_play_knight():
	from game.pytan.__init__ import PyTan

	def callback_play_knight(self: PyTan, player_index):
		if not self.player_rolled:
			return False
		if not self.get_player_is_current_player(player_index):
			return False
		if not self.get_private_development_card_count(player_index, game_settings.knight_index) > 0:
			return False
		return True

	return callback_play_knight
