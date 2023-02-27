from pytan_fast import settings


def handle_offer_player_trade(trade, player_index, self):
	self.set_player_offering_trade(player_index, 1)
	self.current_trades_turn += 1
	for opponent_index in settings.player_list:
		self.set_player_accepted_trade(opponent_index, 0)
		self.set_player_declined_trade(opponent_index, 0)
	for resource_index in settings.resource_list:
		self.set_current_player_trade(resource_index, trade[resource_index])
	self.current_player_trade = trade
	self.player_trade_block_state = True
