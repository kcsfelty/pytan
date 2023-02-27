from pytan_fast import settings

def handle_confirm_player_trade(player_index, trade_partner_index, self):
	self.player_trade_block_state = False
	self.ever_confirmed_player_trade = True
	trade = [0 for _ in settings.resource_list]
	for resource_index in settings.resource_list:
		change = self.get_current_player_trade(resource_index)
		trade[resource_index] = change
		self.set_current_player_trade(resource_index, 0)
	self.trade_players(trade_partner_index, player_index, trade)
	self.current_player_trade = None
	for resource_index in settings.resource_list:
		self.set_current_player_trade(resource_index, 0)
	self.player_trade_block_state = False

	for opponent_index in settings.player_list:
		self.set_player_offering_trade(opponent_index, 0)
		self.set_player_accepted_trade(opponent_index, 0)
		self.set_player_declined_trade(opponent_index, 0)


