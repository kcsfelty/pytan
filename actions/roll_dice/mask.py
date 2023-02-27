

def mask_roll_dice():
	from game.pytan.__init__ import PyTan

	def callback_roll_dice(self: PyTan, player_index):
		if self.player_rolled:
			return False
		if not self.get_player_is_current_player(player_index):
			return False
		return True

	return callback_roll_dice
