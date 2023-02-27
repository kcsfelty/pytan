# from game.game_settings import development_card_cost
#
#
# def mask_buy_development_card():
# 	from game.pytan.__init__ import PyTan
#
# 	def callback_buy_development_card(self: PyTan, player_index):
# 		if not self.get_player_is_current_player(player_index):
# 			return False
# 		if not self.player_rolled:
# 			return False
# 		if not self.get_bank_development_card_count() > 0:
# 			return False
# 		if not self.player_can_afford(player_index, development_card_cost):
# 			return False
# 		return True
#
# 	return callback_buy_development_card
