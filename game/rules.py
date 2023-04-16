import numpy as np
import reference.settings as gs


max_cards = np.array([19, 19, 19, 19, 19])
min_cards = np.array([0, 0, 0, 0, 0])


class Rules:
	def __init__(self, game):
		self.game = game

	def resource_bounds(self, hand, change=min_cards):
		hand += change
		return ((min_cards <= hand) & (hand <= max_cards)).any()

	def bank_resource_bounds(self, game_index, change=min_cards):
		return self.resource_bounds(self.game.state.bank_resources[game_index], change)

	def player_resource_bounds(self, player, game_index, change=min_cards):
		return self.resource_bounds(player.resource_cards[game_index], change)

	def game_resource_bounds(self, game_index):
		card_sum = np.zeros_like(gs.resource_card_count_per_type)
		card_sum += self.game.state.bank_resources[game_index]
		for player in self.game.player_list:
			card_sum += player.resource_cards[game_index]
		return self.resource_bounds(card_sum)

	def development_bounds(self, hand, change=min_cards):
		hand += change
		return ((min_cards <= hand) & (hand <= gs.development_card_count_per_type)).any()

	def get_bank_development(self, game_index):
		return np.histogram(self.game.development_card_stack[game_index], bins=[0, 1, 2, 3, 4, 5])

	def bank_development_bounds(self, game_index, change=min_cards):
		return self.development_bounds(self.get_bank_development(game_index), change)

	def player_played_development_bounds(self, player, game_index, change=min_cards):
		return self.development_bounds(player.development_cards_played[game_index], change)

	def player_bought_development_bounds(self, player, game_index, change=min_cards):
		return self.development_bounds(player.development_card_bought_this_turn[game_index], change)

	def player_hand_development_bounds(self, player, game_index, change=min_cards):
		return self.development_bounds(player.development_cards[game_index], change)

	def get_player_sum_development(self, player, game_index, change=min_cards):
		card_sum = np.zeros_like(gs.development_card_count_per_type)
		card_sum += change
		card_sum += player.development_cards_played[game_index]
		card_sum += player.development_card_bought_this_turn[game_index]
		card_sum += player.development_cards[game_index]
		return card_sum

	def game_development_bounds(self, game_index):
		card_sum = np.zeros_like(gs.development_card_count_per_type)
		card_sum += self.get_bank_development(game_index)
		for player in self.game.player_list:
			card_sum += self.get_player_sum_development(player, game_index)
		return self.development_bounds(card_sum)

	def player_sum_development_bounds(self, player, game_index):
		return self.development_bounds(self.get_player_sum_development(player, game_index))

	def player_distinct_development_bound(self, player, game_index):
		played = self.player_played_development_bounds(player, game_index)
		bought = self.player_bought_development_bounds(player, game_index)
		hand = self.player_hand_development_bounds(player, game_index)
		return played and bought and hand

	def player_development_bounds(self, player, game_index):
		distinct = self.player_distinct_development_bound(player, game_index)
		dev_sum = self.player_sum_development_bounds(player, game_index)
		return distinct and dev_sum

	def settlement_bounds(self, player, game_index, change=0):
		count = change
		count += player.settlement_count[game_index]
		min_count = 0 <= count
		max_count = count <= gs.max_settlement_count
		return min_count and max_count

	def city_bounds(self, player, game_index, change=0):
		count = change
		count += player.city_count[game_index]
		min_count = 0 <= count
		max_count = count <= gs.max_city_count
		return min_count and max_count

	def road_bounds(self, player, game_index, change=0):
		count = change
		count += player.road_count[game_index]
		min_count = 0 <= count
		max_count = count <= gs.max_road_count
		return min_count and max_count

	def building_bounds(self, player, game_index):
		settlement = self.settlement_bounds(player, game_index)
		city = self.city_bounds(player, game_index)
		road = self.road_bounds(player, game_index)
		return settlement and city and road

	def player_bounds(self, player, game_index):
		resource = self.player_resource_bounds(player, game_index)
		development = self.player_development_bounds(player, game_index)
		building = self.building_bounds(player, game_index)
		return resource and development and building

	def bank_bounds(self, game_index):
		resource = self.bank_resource_bounds(game_index)
		development = self.bank_development_bounds(game_index)
		return resource and development

	def game_bounds(self, game_index):
		valid = self.bank_bounds(game_index)
		valid = valid and self.game_resource_bounds(game_index)
		valid = valid and self.game_development_bounds(game_index)
		for player in self.game.player_list:
			valid = valid and self.player_bounds(player, game_index)
		return valid

	def current_player(self, player, game_index):
		valid = player.current_player[game_index]
		valid = valid and player is self.game.current_player[game_index]
		return valid

	def dice_rolled(self, game_index):
		return np.any(self.game.state.current_roll[game_index] == 1)

	def build_phase(self, game_index):
		return self.game.state.build_phase[game_index]

	def post_roll(self, player, game_index):
		valid = self.current_player(player, game_index)
		valid = valid and not self.build_phase(game_index)
		valid = valid and self.dice_rolled(game_index)
		return valid

	def pre_roll(self, player, game_index):
		valid = self.current_player(player, game_index)
		valid = valid and not self.build_phase(game_index)
		valid = valid and not self.dice_rolled(game_index)
		return valid

	def player_trade(self, player_from, player_to, trade, game_index):
		valid = not self.build_phase(game_index)
		valid = valid and self.current_player(player_from, game_index)
		valid = valid and not self.current_player(player_to, game_index)
		valid = valid and player_from.can_afford(trade, game_index)
		valid = valid and player_to.can_afford(-trade, game_index)
		return valid

	def decline_player_trade(self, player, game_index):
		offered_player = self.game.trading_player[game_index]
		trade_is_offered = np.sum(self.game.state.current_player_trade[game_index] != 0) > 0
		valid = not self.build_phase(game_index)
		valid = valid and self.dice_rolled(game_index)
		valid = valid and not self.current_player(player, game_index)
		valid = valid and not player.offering_trade[game_index]
		valid = valid and self.current_player(offered_player, game_index)
		valid = valid and offered_player.offering_trade[game_index]
		valid = valid and trade_is_offered
		return valid

	def accept_player_trade(self, player, game_index):
		offered_trade = self.game.state.current_player_trade[game_index]
		valid = self.decline_player_trade(player, game_index)
		valid = valid and player.can_afford(-offered_trade, game_index)
		return valid

	def cancel_player_trade(self, player, game_index):
		offered_player = self.game.trading_player[game_index]
		trade_is_offered = np.sum(self.game.state.current_player_trade[game_index] != 0) > 0
		valid = not self.build_phase(game_index)
		valid = valid and self.dice_rolled(game_index)
		valid = valid and player.offering_trade[game_index]
		valid = valid and offered_player is player
		valid = valid and trade_is_offered
		waiting_on_trade_responses = False
		for opponent in self.game.trading_player[game_index].other_players:
			if not (opponent.accepted_trade[game_index] or opponent.declined_trade[game_index]):
				waiting_on_trade_responses = True
		valid = valid and not waiting_on_trade_responses
		return valid

	def confirm_player_trade(self, trade_player, player, game_index):
		offered_trade = self.game.state.current_player_trade[game_index]
		valid = self.cancel_player_trade(player, game_index)
		valid = valid and player.can_afford(offered_trade, game_index)
		valid = valid and self.current_player(player, game_index)
		valid = valid and not self.current_player(trade_player, game_index)
		valid = valid and trade_player is not player
		valid = valid and trade_player.can_afford(-offered_trade, game_index)
		return valid

	def end_turn_build_phase(self, player, game_index):
		valid = self.current_player(player, game_index)
		valid = valid and self.game.state.build_phase_placed_settlement[game_index]
		valid = valid and self.game.state.build_phase_placed_road[game_index]
		return valid

	def end_turn_normal_phase(self, player, game_index):
		valid = self.current_player(player, game_index)
		valid = valid and self.dice_rolled(game_index)
		return valid

	def end_turn(self, player, game_index):
		if self.build_phase(game_index):
			return self.end_turn_build_phase(player, game_index)
		return self.end_turn_normal_phase(player, game_index)

	def owns_tile_vertex(self, player, tile, game_index):
		for adjacent_vertex in tile.vertices:
			if adjacent_vertex.owned_by[game_index] == player:
				return True
		return False

	def player_can_be_robbed(self, player, game_index):
		rob_tile = self.game.board.robbed_tile[game_index]
		valid = not self.current_player(player, game_index)
		valid = valid and not self.build_phase(game_index)
		valid = valid and (player.resource_card_count[game_index] > 0).all()
		valid = valid and self.owns_tile_vertex(player, rob_tile, game_index)
		return valid

	def validate_initial_board(self, game_index):
		if self.game.state.bank_development_card_count[game_index] != 25:
			return False
		if self.game.state.longest_road_length[game_index] != 0:
			return False
		if self.game.state.largest_army_size[game_index] != 0:
			return False
		if not self.game.state.build_phase[game_index]:
			return False
		if self.game.state.build_phase_reversed[game_index]:
			return False
		if self.game.state.build_phase_placed_settlement[game_index]:
			return False
		if self.game.state.build_phase_placed_road[game_index]:
			return False
		if self.game.state.bought_development_card_count[game_index]:
			return False
		if self.game.state.played_development_card_count[game_index]:
			return False
		if not (self.game.state.vertex_settlement[game_index] == np.zeros_like(self.game.state.vertex_settlement[game_index])).all():
			return False
		if not (self.game.state.vertex_city[game_index] == np.zeros_like(self.game.state.vertex_city[game_index])).all():
			return False
		if not (self.game.state.vertex_city[game_index] != np.zeros_like(self.game.state.vertex_city[game_index])).all():
			return False
		if not (self.game.state.edge_open[game_index] != np.zeros_like(self.game.state.edge_open[game_index])).all():
			return False
		if np.sum(self.game.state.tile_has_robber[game_index]) != 1:
			return False
		if self.game.state.bank_resources[game_index] != [19, 19, 19, 19, 19]:
			return False
		if (self.game.state.current_player_trade[game_index] != 0).any():
			return False
		if (self.game.state.current_roll != 0).any():
			return False
		for tile in self.game.board.tiles:
			if np.sum(tile.resource[game_index]) != 1:
				return False
			if np.sum(tile.roll_number[game_index]) != 1:
				return False
			if tile.has_robber[game_index] and tile.resource[game_index] != [0, 0, 0, 0, 0, 1]:
				return False
			if tile.players_to_rob[game_index] is not np.zeros_like(tile.players_to_rob[game_index]):
				return False
		for vertex in self.game.board.vertices:
			if vertex.settlement[game_index]:
				return False
			if vertex.city[game_index]:
				return False
			if not vertex.open[game_index]:
				return False
			if vertex.owned_by[game_index]:
				return False
		for edge in self.game.board.edges:
			if not edge.open:
				return False
		for player in self.game.player_list:
			if player.must_move_robber[game_index]:
				return False
			if player.must_discard[game_index]:
				return False
			if player.victory_points[game_index]:
				return False
			if player.actual_victory_points[game_index]:
				return False
			if (player.resource_cards[game_index]).any():
				return False
			if player.resource_card_count[game_index]:
				return False
			if player.development_cards[game_index].any():
				return False
			if player.development_card_count[game_index]:
				return False
			if player.development_cards_played[game_index].any():
				return False
			if player.settlement_count[game_index]:
				return False
			if player.city_count[game_index]:
				return False
			if player.road_count[game_index]:
				return False
			if player.longest_road[game_index]:
				return False
			if player.owns_longest_road[game_index]:
				return False
			if player.owns_largest_army[game_index]:
				return False
			if player.offering_trade[game_index]:
				return False
			if player.accepted_trade[game_index]:
				return False
			if player.declined_trade[game_index]:
				return False
			if player.must_discard[game_index] != 0:
				return False
			if player.edge_list[game_index]:
				return False
			if player.edge_proximity_vertices[game_index]:
				return False



		for roll_count, roll_tiles in zip(gs.tile_roll_number_count_per_type, self.game.board.roll_hash[game_index]):
			if roll_count is not len(roll_tiles):
				return False
		resource_counts = np.zeros_like(gs.tile_resource_count, dtype=np.int32)
		for tile in self.game.board.tiles:
			resource_counts += tile.resource[game_index]










