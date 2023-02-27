from pytan_fast import settings


def handle_play_monopoly(resource_index, player_index, self):
	self.play_development_card(player_index, settings.monopoly_index)
	card_total = 0
	for opponent_index in settings.player_list:
		if opponent_index is not player_index:
			opponent_has = self.get_private_resource_count(opponent_index, resource_index)
			card_total += opponent_has
			self.set_private_resource_count(opponent_index, resource_index, 0)
	self.set_private_resource_count(player_index, resource_index, card_total)
	self.calculate_cards()