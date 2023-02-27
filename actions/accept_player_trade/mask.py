def mask_accept_player_trade():
	from pytan_fast.game import PyTanFast

	def callback_accept_player_trade(self: PyTanFast, player_index):
		if not self.current_player_trade:
			return False
		if self.get_player_accepted_trade(player_index):
			return False
		if self.get_player_declined_trade(player_index):
			return False
		if not self.player_can_afford(player_index, [-1 * resource for resource in self.current_player_trade]):
			return False
		return True

	return callback_accept_player_trade
