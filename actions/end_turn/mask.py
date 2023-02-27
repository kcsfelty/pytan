

def mask_end_turn():
	from game.pytan.__init__ import PyTan

	def callback_end_turn(self: PyTan, player_index):
		if not self.get_player_is_current_player(player_index):
			return False
		return True

	return callback_end_turn
