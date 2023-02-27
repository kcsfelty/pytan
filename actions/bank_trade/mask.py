
def mask_bank_trade(trade):
	from game.pytan.__init__ import PyTan

	def callback_bank_trade(self: PyTan, player_index):
		if not self.player_rolled:
			return False
		if not self.get_player_is_current_player(player_index):
			return False
		if not self.player_can_afford(player_index, trade):
			return False
		return True

	return callback_bank_trade
