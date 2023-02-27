from game import game_settings


def mask_cancel_player_trade():
	from game.pytan.__init__ import PyTan

	def callback_cancel_player_trade(self: PyTan, player_index):
		if not self.get_player_offering_trade(player_index):
			return False
		for opponent_index in game_settings.player_list:
			if opponent_index is not player_index:
				if not (self.get_player_accepted_trade(opponent_index) or self.get_player_declined_trade(opponent_index)):
					return False
		return True

	return callback_cancel_player_trade
