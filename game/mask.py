import numpy as np
import reference.definitions as df
from reference.action_mapping import action_mapping


class Mask:
	def __init__(self, game_count=1):
		self.mask = np.zeros((game_count, sum([action_mapping[term] for term in action_mapping])), dtype=np.int32)

		self.mask_slices = {}
		current_index = 0
		for term in action_mapping:
			self.mask_slices[term] = self.mask[:, current_index:current_index + action_mapping[term]]
			current_index += action_mapping[term]

		self.accept_player_trade = self.mask_slices[df.accept_player_trade]
		self.bank_trade = self.mask_slices[df.bank_trade]
		self.buy_development_card = self.mask_slices[df.buy_development_card]
		self.cancel_player_trade = self.mask_slices[df.cancel_player_trade]
		self.confirm_player_trade = self.mask_slices[df.confirm_player_trade]
		self.decline_player_trade = self.mask_slices[df.decline_player_trade]
		self.discard = self.mask_slices[df.discard]
		self.end_turn = self.mask_slices[df.end_turn]
		self.general_port_trade = self.mask_slices[df.general_port_trade]
		self.move_robber = self.mask_slices[df.move_robber]
		self.no_action = self.mask_slices[df.no_action]
		self.offer_player_trade = self.mask_slices[df.offer_player_trade]
		self.place_city = self.mask_slices[df.place_city]
		self.place_road = self.mask_slices[df.place_road]
		self.place_settlement = self.mask_slices[df.place_settlement]
		self.play_knight = self.mask_slices[df.play_knight]
		self.play_monopoly = self.mask_slices[df.play_monopoly]
		self.play_road_building = self.mask_slices[df.play_road_building]
		self.play_year_of_plenty = self.mask_slices[df.play_year_of_plenty]
		self.resource_port_trade = self.mask_slices[df.resource_port_trade]
		self.rob_player = self.mask_slices[df.rob_player]
		self.roll_dice = self.mask_slices[df.roll_dice]

	def only(self, term, game_index):
		self.reset(game_index)
		self.mask_slices[term][game_index].fill(True)

	def can(self, term, game_index):
		self.mask_slices[term][game_index].fill(True)

	def cannot(self, term, game_index):
		self.mask_slices[term][game_index].fill(False)

	def reset(self, game_index):
		self.mask[game_index].fill(False)
