

def mask_move_robber(tile):
	from game.pytan.__init__ import PyTan

	def callback_move_robber(self: PyTan, player_index):
		if self.robber_position is tile:
			return False
		if not self.get_player_must_move_robber(player_index):
			return False
		return True

	return callback_move_robber
