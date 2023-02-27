from game import game_settings


def mask_offer_player_trade(trade):
	from game.pytan.__init__ import PyTan

	def callback_offer_player_trade(self: PyTan, player_index):
		if self.current_trades_turn == self.maximum_trades_per_turn:
			return False
		if not self.get_player_is_current_player(player_index):
			return False
		if not self.player_can_afford(player_index, trade):
			return False
		if self.get_player_offering_trade(player_index):
			return False
		if not self.current_player_trade:
			reverse_trade = [-1 * resource for resource in trade]
			trade_partner_can_afford = [self.player_can_afford(trade_partner_index, reverse_trade) for trade_partner_index in game_settings.player_list]
			trade_partner_can_afford.pop(player_index)
			if True not in trade_partner_can_afford:
				return False
		return True

	return callback_offer_player_trade
