

def mask_no_action():
	from game.pytan.__init__ import PyTan

	def callback_no_action(self: PyTan, player_index):
		if self.get_player_is_current_player(player_index):
			return False
		return True

	return callback_no_action
