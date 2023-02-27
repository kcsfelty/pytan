from pytan_fast import settings


def handle_cancel_player_trade(_, player_index, self):
	self.current_player_trade = None
	for resource_index in settings.resource_list:
		self.set_current_player_trade(resource_index, 0)
	self.player_trade_block_state = False
	self.set_player_offering_trade(player_index, 0)
	for opponent_index in settings.player_list:
		self.set_player_accepted_trade(opponent_index, 0)
		self.set_player_declined_trade(opponent_index, 0)
