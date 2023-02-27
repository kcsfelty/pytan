

def mask_rob_player(rob_player_index):
	from game.pytan.__init__ import PyTan

	def callback_rob_player(self: PyTan, player_index):
		if not self.get_player_is_current_player(player_index):
			return False
		if rob_player_index not in self.players_to_rob:
			return False
		return True

	return callback_rob_player
