import random

import numpy as np

import pytan_fast.definitions as df
import pytan_fast.settings as gs
from pytan_fast.get_trades import get_trades
from util.reverse_histogram import reverse_histogram


def get_trade_lookup(trade_list):
	return np.all(np.mgrid[0:5, 0:5, 0:5, 0:5, 0:5].T + np.expand_dims(trade_list, axis=(1, 2, 3, 4, 5)) >= 0, axis=-1).T

no_cost = np.zeros(5)
max_cards = np.zeros(5)
max_cards.fill(-19)

class Handler:
	def __init__(self, game):
		self.game = game
		self.bank_trades, self.general_port_trades, self.resource_port_trades, self.player_trades, self.year_of_plenty_trades, self.discard_trades = get_trades()
		self.bank_trades = np.array(self.bank_trades)
		self.general_port_trades = np.array(self.general_port_trades)
		self.resource_port_trades = np.array(self.resource_port_trades)
		self.player_trades = np.array(self.player_trades)
		self.year_of_plenty_trades = np.array(self.year_of_plenty_trades)

		self.bank_trades_lookup = get_trade_lookup(self.bank_trades)
		self.general_port_trades_lookup = get_trade_lookup(self.general_port_trades)
		self.resource_port_trades_lookup = get_trade_lookup(self.resource_port_trades)
		self.player_trades_lookup = get_trade_lookup(self.player_trades)
		self.discard_trades_lookup = get_trade_lookup(self.discard_trades)

		self.action_lookup = [
			*[(self.handle_end_turn, None)],
			*[(self.handle_roll_dice, None)],
			*[(self.handle_bank_trade, trade) for trade in self.bank_trades],
			*[(self.handle_bank_trade, trade) for trade in self.general_port_trades],
			*[(self.handle_bank_trade, trade) for trade in self.resource_port_trades],
			*[(self.handle_offer_player_trade, trade) for trade in self.player_trades],
			*[(self.handle_accept_player_trade, None)],
			*[(self.handle_decline_player_trade, None)],
			*[(self.handle_cancel_player_trade, None)],
			*[(self.handle_confirm_player_trade, player) for player in self.game.player_list],
			*[(self.handle_move_robber, tile) for tile in self.game.board.tiles],
			*[(self.handle_rob_player, player) for player in self.game.player_list],
			*[(self.handle_discard, trade) for trade in self.discard_trades],
			*[(self.handle_place_city, vertex) for vertex in self.game.board.vertices],
			*[(self.handle_place_road, edge) for edge in self.game.board.edges],
			*[(self.handle_place_settlement, vertex) for vertex in self.game.board.vertices],
			*[(self.handle_buy_development_card, None)],
			*[(self.handle_play_knight, None)],
			*[(self.handle_play_monopoly, resource_index) for resource_index in gs.resource_list],
			*[(self.handle_play_road_building, None)],
			*[(self.handle_play_year_of_plenty, trade) for trade in self.year_of_plenty_trades],
			*[(self.handle_no_action, None)],
		]

	def check_bank_trades(self, player):
		resource_tuple = tuple(np.minimum(player.resource_cards, 4))
		np.copyto(player.dynamic_mask.bank_trade, self.bank_trades_lookup[resource_tuple])
		if player.port_access[5]:
			np.copyto(player.dynamic_mask.general_port_trade, self.general_port_trades_lookup[resource_tuple])
		resource_tuple = tuple(np.minimum(player.resource_cards, 2) * player.port_access[:5])
		np.copyto(player.dynamic_mask.resource_port_trade, self.resource_port_trades_lookup[resource_tuple])

	def check_player_trades(self, player):
		if self.game.player_trades_this_turn == gs.player_trade_per_turn_limit:
			player.dynamic_mask.offer_player_trade.fill(False)
			return
		resource_tuple = tuple(np.minimum(player.resource_cards, 4))
		np.copyto(player.dynamic_mask.offer_player_trade, self.player_trades_lookup[resource_tuple])

	def check_discard_trades(self, player):
		resource_tuple = tuple(np.minimum(player.resource_cards, 4))
		np.copyto(player.dynamic_mask.discard, self.discard_trades_lookup[resource_tuple])

	def check_player_purchase(self, player):
		if player.can_afford(gs.road_cost):
			player.dynamic_mask.place_road.fill(True)
			player.apply_static(df.place_road)
		if player.can_afford(gs.settlement_cost):
			player.dynamic_mask.place_settlement.fill(True)
			player.apply_static(df.place_settlement)
		if player.can_afford(gs.city_cost):
			player.dynamic_mask.place_city.fill(True)
			player.apply_static(df.place_city)
		if self.game.state.game_state_slices[df.bank_development_card_count] > 0:
			player.dynamic_mask.buy_development_card.fill(player.can_afford(gs.development_card_cost))

	def check_player_play_development_card(self, player):
		if self.game.state.game_state_slices[df.played_development_card_count] == gs.max_development_cards_played_per_turn:
			return
		player.dynamic_mask.play_knight.fill(player.development_cards[gs.knight_index] > 0)
		player.dynamic_mask.play_monopoly.fill(player.development_cards[gs.monopoly_index] > 0)
		player.dynamic_mask.play_road_building.fill(player.development_cards[gs.road_building_index] > 0)
		player.dynamic_mask.play_year_of_plenty.fill(player.development_cards[gs.year_of_plenty_index] > 0)

	def check_bank_can_disburse(self, trade):
		single_recipient = np.where(np.sum(np.where(trade, 1, 0), axis=0) == 1, True, False)
		bank_can_afford = np.all(self.game.state.game_state_slices[df.bank_resources] + np.expand_dims(np.sum(trade, axis=0) * -1, axis=0) >= 0, axis=0)
		return np.minimum(np.where(np.logical_or(single_recipient, bank_can_afford), trade, 0), self.game.state.game_state_slices[df.bank_resources])

	def set_post_roll_mask(self, player):
		player.dynamic_mask.only(df.end_turn)
		self.check_bank_trades(player)
		self.check_player_trades(player)
		self.check_player_purchase(player)
		self.check_player_play_development_card(player)
		if self.game.player_trades_this_turn == gs.player_trade_per_turn_limit:
			player.dynamic_mask.offer_player_trade.fill(0)

	def can_afford(self, player, trade):
		return np.all(player.resource_cards + trade >= 0)

	def compare_mask(self, mask):
		for act, mask_value, i in zip(self.action_lookup, mask, range(len(self.action_lookup))):
			callback, args = act
			print(i, callback.__name__, args, mask_value)

	def handle_action(self, action_step, player):
		# print(action_step.action.numpy()[0])
		callback, args = self.action_lookup[action_step.action.numpy()[0]]
		# print(self.game.num_step, str(player).ljust(60), str(action_step.action.numpy()).ljust(10), callback.__name__.ljust(35), str(args).ljust(20), np.sum(player.dynamic_mask.mask))
		callback(args, player)

	def handle_end_turn(self, _, player):
		next_player = next(self.game.player_cycle)
		self.game.current_player = next_player
		player.current_player.fill(1)
		player.development_cards += player.development_card_bought_this_turn
		player.development_card_bought_this_turn.fill(0)
		player.dynamic_mask.only(df.no_action)
		self.game.player_trades_this_turn = 0
		self.game.state.game_state_slices[df.turn_number] += 1
		self.game.current_player = next_player
		next_player.current_player.fill(1)
		next_player.dynamic_mask.only(df.roll_dice)
		self.check_player_play_development_card(next_player)
		self.game.state.game_state_slices[df.bought_development_card_count].fill(0)

	def handle_roll_dice(self, roll, player):
		roll = roll or self.game.dice.roll()
		self.game.state.game_state_slices[df.current_roll].fill(0)
		self.game.state.game_state_slices[df.current_roll][roll].fill(1)
		if roll == gs.robber_activated_on_roll:
			self.handle_robber_roll(player)
			return
		self.handle_distribute_resources(roll, player)
		self.set_post_roll_mask(player)

	def handle_distribute_resources(self, roll, player):
		trade = np.zeros((gs.player_count, gs.resource_type_count), dtype=np.int32)
		for tile in self.game.board.roll_hash[roll]:
			for vertex in tile.vertices:
				if vertex.owned_by:
					trade[vertex.owned_by.index][tile.resource_index] += 1 if vertex.settlement else 2 if vertex.city else 0
		trade = self.check_bank_can_disburse(trade)
		for disburse_player in self.game.player_list:
			self.handle_bank_trade(trade[disburse_player.index], disburse_player)

	def handle_robber_roll(self, player):
		for robbed_player in self.game.player_list:
			robbed_player.dynamic_mask.only(df.no_action)
			if robbed_player.resource_card_count > gs.rob_player_above_card_count:
				robbed_player.dynamic_mask.only(df.discard)
				self.check_discard_trades(robbed_player)
				self.game.immediate_play.insert(0, robbed_player)
				robbed_player.must_discard = robbed_player.resource_card_count // 2
		player.dynamic_mask.only(df.move_robber)
		self.game.immediate_play.append(player)

	def handle_player_trade(self, player_from, player_to, trade):
		player_from.resource_cards += trade
		player_to.resource_cards -= trade
		player_from.resource_card_count = np.sum(player_from.resource_cards)
		player_to.resource_card_count = np.sum(player_to.resource_cards)
		if not self.game.state.game_state_slices[df.build_phase]:
			self.set_post_roll_mask(self.game.current_player)

		# assert player_from.can_afford(no_cost)
		# assert player_to.can_afford(no_cost)
		# assert not player_from.can_afford(max_cards)
		# assert not player_to.can_afford(max_cards)

	def handle_bank_trade(self, trade, player):
		player.resource_cards += trade
		self.game.state.game_state_slices[df.bank_resources] -= trade
		player.resource_card_count = np.sum(player.resource_cards)
		if not self.game.state.game_state_slices[df.build_phase]:
			if player == self.game.current_player:
				self.set_post_roll_mask(player)
		# assert player.can_afford(no_cost)
		# assert not player.can_afford(max_cards)

	def handle_offer_player_trade(self, trade, player):
		self.game.player_trades_this_turn += 1
		player.dynamic_mask.only(df.cancel_player_trade)
		for opponent in player.other_players:
			opponent.dynamic_mask.only(df.decline_player_trade)
			if opponent.can_afford(trade * -1):
				opponent.dynamic_mask.can(df.accept_player_trade)
		self.game.state.game_state_slices[df.current_player_trade] += trade
		self.game.trading_player = player
		self.game.immediate_play.extend(player.other_players)
		self.game.immediate_play.append(player)

	def handle_accept_player_trade(self, _, player):
		player.accepted_trade.fill(1)
		player.dynamic_mask.only(df.no_action)
		self.game.trading_player.dynamic_mask.confirm_player_trade[player.index] = True

	def handle_decline_player_trade(self, _, player):
		player.declined_trade.fill(1)
		player.dynamic_mask.only(df.no_action)

	def handle_cancel_player_trade(self, _, player):
		self.game.trading_player = None
		player.offering_trade.fill(0)
		for opponent in player.other_players:
			opponent.accepted_trade.fill(0)
			opponent.declined_trade.fill(0)
		self.game.state.game_state_slices[df.current_player_trade].fill(0)
		self.set_post_roll_mask(player)

	def handle_confirm_player_trade(self, trade_player, player):
		self.handle_player_trade(player, trade_player, self.game.state.game_state_slices[df.current_player_trade])
		self.handle_cancel_player_trade(None, player)

	def handle_move_robber(self, tile, player):
		player.must_move_robber.fill(0)
		player.dynamic_mask.mask.fill(False)
		self.game.board.robbed_tile.has_robber.fill(0)
		self.game.board.robbed_tile = tile
		self.game.board.robbed_tile.has_robber.fill(1)
		players_to_rob = 0
		for rob_player in self.game.player_list:
			if tile.players_to_rob[player.index, rob_player.index]:
				if np.sum(rob_player.resource_cards) > 0:
					if rob_player.index is not player.index:
						player.dynamic_mask.rob_player[rob_player.index] = True
						players_to_rob += 1
		if players_to_rob >= 1:
			self.game.immediate_play.append(player)
		else:
			self.set_post_roll_mask(player)

	def handle_rob_player(self, rob_player, player):
		rob_player_deck = reverse_histogram(rob_player.resource_cards)
		random.shuffle(rob_player_deck)
		trade = np.zeros(gs.resource_type_count, dtype=np.int32)
		trade[rob_player_deck[0]] = -1 * gs.robber_steals_card_quantity
		self.handle_player_trade(rob_player, player, trade)
		self.set_post_roll_mask(player)

	def handle_discard(self, trade, player):
		self.handle_bank_trade(trade, player)
		player.must_discard -= 1
		if player.must_discard > 0:
			self.check_discard_trades(player)
			self.game.immediate_play.insert(0, player)
			return
		for game_player in self.game.player_list:
			if not game_player.must_discard == 0:
				return
		self.game.immediate_play.append(self.game.current_player)
		self.game.current_player.dynamic_mask.only(df.move_robber)

	def handle_place_city(self, vertex, player):
		self.handle_bank_trade(gs.city_cost, player)
		player.city_count += 1
		vertex.city.fill(1)
		player.settlement_count -= 1
		vertex.settlement.fill(0)
		player.victory_points += gs.city_victory_points
		player.actual_victory_points += gs.city_victory_points
		if player.city_count == gs.max_city_count:
			player.static_mask.place_city.fill(0)

	def handle_place_road(self, edge, player):
		for block_player in self.game.player_list:
			block_player.static_mask.place_road[self.game.board.edge_hash[edge.index]] = 0
		player.road_count += 1
		edge.open.fill(0)
		player.owned_edges[self.game.board.edge_hash[edge.index]] = 1
		player.edge_list.append(edge)
		if self.game.state.game_state_slices[df.build_phase]:
			self.game.state.game_state_slices[df.build_phase_placed_road] = 1
		elif self.game.resolve_road_building_count > 0:
			self.game.resolve_road_building_count -= 1
			if self.game.resolve_road_building_count == 0:
				self.game.immediate_play.append(player)
				self.set_post_roll_mask(player)
		else:
			self.handle_bank_trade(gs.road_cost, player)
		if player.road_count == gs.max_road_count - 1:
			player.static_mask.cannot(df.place_road)
		for adjacent_edge in edge.edges:
			if adjacent_edge.open:
				player.static_mask.place_road[self.game.board.edge_hash[adjacent_edge.index]] = 1
		for adjacent_vertex in edge.vertices:
			if adjacent_vertex.open:
				player.static_mask.place_settlement[self.game.board.vertex_hash[adjacent_vertex.index]] = 1
		player.calculate_longest_road()
		if player.longest_road > self.game.state.game_state_slices[df.longest_road_length]:
			if not player.owns_longest_road:
				if player.longest_road > gs.min_longest_road:
					if self.game.longest_road_owner:
						self.game.longest_road_owner.victory_points -= gs.longest_road_victory_points
						self.game.longest_road_owner.actual_victory_points -= gs.longest_road_victory_points
						self.game.longest_road_owner.owns_longest_road.fill(0)
					np.copyto(self.game.state.game_state_slices[df.longest_road_length], player.longest_road)
					self.game.longest_road_owner = player
					self.game.longest_road_owner.victory_points += gs.longest_road_victory_points
					self.game.longest_road_owner.actual_victory_points += gs.longest_road_victory_points
					self.game.longest_road_owner.owns_longest_road.fill(1)

	def handle_place_settlement(self, vertex, player):
		if self.game.state.game_state_slices[df.build_phase]:
			player.dynamic_mask.cannot(df.place_settlement)
			for edge in vertex.edges:
				player.dynamic_mask.place_road[self.game.board.edge_hash[edge.index]] = True
				player.static_mask.place_road[self.game.board.edge_hash[edge.index]] = True
			self.game.state.game_state_slices[df.build_phase_placed_settlement] = 1
			if self.game.state.game_state_slices[df.build_phase_reversed]:
				trade = np.zeros(gs.resource_type_count_tile, dtype=np.int32)
				for tile in vertex.tiles:
					trade[tile.resource_index] += 1
				self.handle_bank_trade(trade[:5], player)
		else:
			self.handle_bank_trade(gs.settlement_cost, player)
		player.settlement_count += 1
		player.victory_points += gs.settlement_victory_points
		player.actual_victory_points += gs.settlement_victory_points
		np.logical_or(player.port_access, vertex.port, out=player.port_access)
		player.vertex_list.append(vertex)
		player.static_mask.place_city[self.game.board.vertex_hash[vertex.index]] = 1
		vertex.owned_by = player
		vertex.settlement.fill(1)
		vertex.open.fill(0)
		player.owned_vertices[self.game.board.vertex_hash[vertex.index]] = 1
		for block_player in self.game.player_list:
			block_player.static_mask.place_settlement[self.game.board.vertex_hash[vertex.index]] = 0
		for adjacent_vertex in vertex.vertices:
			adjacent_vertex.open.fill(0)
			for block_player in self.game.player_list:
				block_player.static_mask.place_settlement[self.game.board.vertex_hash[vertex.index]] = 0
		for adjacent_tile in vertex.tiles:
			adjacent_tile.players_to_rob[:, player.index] = True
			adjacent_tile.players_to_rob[player.index, player.index] = False

	def handle_buy_development_card(self, _, player):
		self.handle_bank_trade(gs.development_card_cost, player)
		card_index = self.game.development_card_stack.pop(0)
		self.game.state.game_state_slices[df.bank_development_card_count] = len(self.game.development_card_stack)
		self.game.state.game_state_slices[df.bought_development_card_count] += 1
		if self.game.state.game_state_slices[df.bank_development_card_count] == 0:
			for player in self.game.player_list:
				player.static_mask.cannot(df.buy_development_card)
		if card_index == gs.victory_point_card_index:
			player.development_cards[card_index] += 1
		else:
			player.development_card_bought_this_turn[card_index] += 1

	def handle_play_dev_card(self, card_index, player):
		player.development_cards[card_index] -= 1
		player.development_cards_played[card_index] += 1
		player.development_card_count -= 1
		self.game.state.game_state_slices[df.played_development_card_count] += 1

	def handle_play_knight(self, _, player):
		self.handle_play_dev_card(gs.knight_index, player)
		self.game.immediate_play.append(player)
		player.dynamic_mask.only(df.move_robber)
		player.must_move_robber.fill(1)
		if player.development_cards_played[gs.knight_index] >= gs.min_largest_army:
			if player.development_cards_played[gs.knight_index] > self.game.state.game_state_slices[df.largest_army_size]:
				if self.game.largest_army_owner:
					self.game.largest_army_owner.owns_largest_army.fill(0)
					self.game.largest_army_owner.victory_points -= gs.largest_army_victory_points
					self.game.largest_army_owner.actual_victory_points -= gs.largest_army_victory_points
				if not player.owns_largest_army:
					player.owns_largest_army.fill(1)
					player.victory_points += gs.largest_army_victory_points
					player.actual_victory_points += gs.largest_army_victory_points

	def handle_play_monopoly(self, resource_index, player):
		self.handle_play_dev_card(gs.monopoly_index, player)
		for opponent in player.other_players:
			trade = np.zeros(gs.resource_type_count, dtype=np.int32)
			count = opponent.resource_cards[resource_index]
			trade[resource_index] = -1 * count
			self.handle_player_trade(opponent, player, trade)

	def handle_play_road_building(self, _, player):
		self.handle_play_dev_card(gs.road_building_index, player)
		self.game.immediate_play.append(player)
		self.game.immediate_play.append(player)
		self.game.resolve_road_building_count = gs.road_building_road_count
		player.dynamic_mask.only(df.place_road)

	def handle_play_year_of_plenty(self, trade, player):
		self.handle_play_dev_card(gs.year_of_plenty_index, player)
		self.handle_bank_trade(trade, player)

	def handle_no_action(self, _, player):
		pass
