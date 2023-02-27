from pytan_fast import settings
from util.pick_random_weighted_index import pick_random_weighted_index


bought_victory_point_card = "BOUGHT_VICTORY_POINT_CARD"


def handle_buy_development_card(_, player_index, self):
	self.trade_with_bank(player_index, settings.development_card_cost)
	development_card_index = int(pick_random_weighted_index(self.bank_development_cards))
	self.bank_development_cards[development_card_index] -= 1
	self.set_bank_development_card_count(sum(self.bank_development_cards))
	if development_card_index == settings.victory_point_card_index:
		change = self.get_private_development_card_count(player_index, development_card_index) + 1
		self.set_private_development_card_count(player_index, development_card_index, change)
		self.calculate_victory_points()
		self.filter_action(player_index, bought_victory_point_card)
	else:
		game_change = self.get_current_player_bought_development_card_count() + 1
		self.set_current_player_bought_development_card_count(game_change)
		private_change = self.get_private_development_card_bought_this_turn_count(player_index, development_card_index) + 1
		self.set_private_development_card_bought_this_turn_count(player_index, development_card_index, private_change)
		public_change = self.get_player_development_card_count(player_index) + 1
		self.set_player_development_card_count(player_index, public_change)

