import numpy as np

from actions.discard.mapping import discard_prefix


class Mask:
	def __init__(
			self,
			player,
			game,
			accept_player_trade,
			bank_trade,
			buy_development_card,
			cancel_player_trade,
			confirm_player_trade,
			decline_player_trade,
			discard,
			end_turn,
			general_port_trade,
			move_robber,
			no_action,
			offer_player_trade,
			place_city,
			place_road,
			place_settlement,
			play_knight,
			play_monopoly,
			play_road_building,
			play_year_of_plenty,
			resource_port_trade,
			rob_player,
			roll_dice
		 ):
		self.player = player
		self.game = game
		self.offer_player_trade = offer_player_trade
		self.accept_player_trade = accept_player_trade
		self.decline_player_trade = decline_player_trade
		self.cancel_player_trade = cancel_player_trade
		self.confirm_player_trade = confirm_player_trade

		self.bank_trade = bank_trade
		self.general_port_trade = general_port_trade
		self.resource_port_trade = resource_port_trade

		self.discard = discard
		self.end_turn = end_turn
		self.roll_dice = roll_dice
		self.move_robber = move_robber
		self.rob_player = rob_player

		self.place_city = place_city
		self.place_road = place_road
		self.place_settlement = place_settlement

		self.buy_development_card = buy_development_card
		self.play_knight = play_knight
		self.play_monopoly = play_monopoly
		self.play_road_building = play_road_building
		self.play_year_of_plenty = play_year_of_plenty

		self.no_action = no_action

	def calculate_trades(self):
		self.bank_trade[:] = np.repeat(np.where(self.player.resource_cards >= 4, 1, 0), 4)
		self.general_port_trade[:] = np.repeat(np.logical_and(np.where(self.player.resource_cards >= 3, 1, 0), self.player.port_access[0, :5]), 4)
		self.resource_port_trade[:] = np.repeat(np.logical_and(np.where(self.player.resource_cards >= 2, 1, 0), self.player.port_access[0, 5]), 4)
		print(self.game.grouped_actions)
		print(self.game.grouped_actions[discard_prefix])
