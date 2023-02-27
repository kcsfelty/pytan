def mask_decline_player_trade():
	from game.pytan.__init__ import PyTan

	def callback_decline_player_trade(self: PyTan, player_index):
		if self.get_player_offering_trade(player_index):
			return False
		if not self.current_player_trade:
			return False
		if self.get_player_accepted_trade(player_index):
			return False
		if self.get_player_declined_trade(player_index):
			return False
		return True

	return callback_decline_player_trade
