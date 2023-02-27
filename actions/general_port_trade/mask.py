from game import game_settings


def mask_general_port_trade(trade):
	from game.pytan.__init__ import PyTan

	def callback_general_port_trade(self: PyTan, player_index):
		if not self.get_player_is_current_player(player_index):
			return False
		if not self.player_rolled:
			return False
		if self.get_build_phase():
			return False
		if not self.player_port_access[player_index][game_settings.general_port_index]:
			return False
		if not self.player_can_afford(player_index, trade):
			return False
		return True

	return callback_general_port_trade
