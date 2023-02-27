from pytan_fast import settings

def handle_discard(trade, player_index, self):
	self.player_discard_count[player_index] -= 1
	self.set_player_resource_card_count(player_index, sum(self.private_observation_players[player_index][:5]))
	player_must_discard = False
	for discard_player_index in settings.player_list:
		if self.player_discard_count[discard_player_index] > 0:
			player_must_discard = True
	self.discard_block_state = player_must_discard
	self.trade_with_bank(player_index, trade)
