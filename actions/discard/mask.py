def mask_discard(trade):
	from game.pytan.__init__ import PyTan

	def callback_discard(self: PyTan, player_index):
		if not self.discard_block_state:
			return False
		if not self.player_discard_count[player_index] > 0:
			return False
		if not self.player_can_afford(player_index, trade):
			return False
		return True

	return callback_discard
