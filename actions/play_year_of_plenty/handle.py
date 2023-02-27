from pytan_fast import settings


def handle_play_year_of_plenty(trade, player_index, self):
	self.play_development_card(player_index, settings.year_of_plenty_index)
	self.trade_with_bank(player_index, trade)

