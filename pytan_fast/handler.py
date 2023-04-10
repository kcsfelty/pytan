import random

import numpy as np

import pytan_fast.definitions as df
import pytan_fast.settings as gs

from pytan_fast.trading import bank_trades, general_port_trades, resource_port_trades, player_trades, \
	discard_trades, year_of_plenty_trades, bank_trades_lookup, bank_trades_lookup_bank, general_port_trades_lookup, \
	general_port_trades_lookup_bank, resource_port_trades_lookup, resource_port_trades_lookup_bank, \
	player_trades_lookup, discard_trades_lookup
from util.reverse_histogram import reverse_histogram


class Handler:
	def __init__(self, game, log_action=False):
		self.game = game
		self.log_action = log_action
		self.action_lookup = [
			*[(self.handle_end_turn, None)],
			*[(self.handle_roll_dice, None)],
			*[(self.handle_bank_trade, trade) for trade in bank_trades],
			*[(self.handle_bank_trade, trade) for trade in general_port_trades],
			*[(self.handle_bank_trade, trade) for trade in resource_port_trades],
			*[(self.handle_offer_player_trade, trade) for trade in player_trades],
			*[(self.handle_accept_player_trade, None)],
			*[(self.handle_decline_player_trade, None)],
			*[(self.handle_cancel_player_trade, None)],
			*[(self.handle_confirm_player_trade, player) for player in self.game.player_list],
			*[(self.handle_move_robber, tile) for tile in self.game.board.tiles],
			*[(self.handle_rob_player, player) for player in self.game.player_list],
			*[(self.handle_discard, trade) for trade in discard_trades],
			*[(self.handle_place_city, vertex) for vertex in self.game.board.vertices],
			*[(self.handle_place_road, edge) for edge in self.game.board.edges],
			*[(self.handle_place_settlement, vertex) for vertex in self.game.board.vertices],
			*[(self.handle_buy_development_card, None)],
			*[(self.handle_play_knight, None)],
			*[(self.handle_play_monopoly, resource_index) for resource_index in gs.resource_list],
			*[(self.handle_play_road_building, None)],
			*[(self.handle_play_year_of_plenty, trade) for trade in year_of_plenty_trades],
			*[(self.handle_no_action, None)],
		]

	def check_bank_trades(self, player, game_index):
		resource_tuple = tuple(np.minimum(player.resource_cards[game_index], 4))
		player_can_afford = bank_trades_lookup[resource_tuple]
		bank_resource_tuple = tuple(np.minimum(self.game.state.bank_resources[game_index], 4))
		bank_can_afford = bank_trades_lookup_bank[bank_resource_tuple]
		np.copyto(
			player.dynamic_mask.mask_slices[df.bank_trade][game_index],
			np.logical_and(
				player_can_afford,
				bank_can_afford))

	def check_general_port_trades(self, player, game_index):
		if player.port_access[game_index][5]:
			resource_tuple = tuple(np.minimum(player.resource_cards[game_index], 4))
			player_can_afford = general_port_trades_lookup[resource_tuple]
			bank_resource_tuple = tuple(np.minimum(self.game.state.bank_resources[game_index], 4))
			bank_can_afford = general_port_trades_lookup_bank[bank_resource_tuple]
			np.copyto(
				player.dynamic_mask.mask_slices[df.general_port_trade][game_index],
				np.logical_and(
					player_can_afford,
					bank_can_afford))

	def check_resource_port_trades(self, player, game_index):
		resource_tuple = tuple(np.minimum(player.resource_cards[game_index], 4) * player.port_access[game_index][:5])
		player_can_afford = resource_port_trades_lookup[resource_tuple]
		bank_resource_tuple = tuple(np.minimum(self.game.state.bank_resources[game_index], 4))
		bank_can_afford = resource_port_trades_lookup_bank[bank_resource_tuple]
		np.copyto(
			player.dynamic_mask.mask_slices[df.resource_port_trade][game_index],
			np.logical_and(
				player_can_afford,
				bank_can_afford))

	def check_player_trades(self, player, game_index):
		if self.game.player_trades_this_turn[game_index] is not gs.player_trade_per_turn_limit:
			resource_tuple = tuple(np.minimum(player.resource_cards[game_index], 4))
			np.copyto(player.dynamic_mask.mask_slices[df.offer_player_trade][game_index], player_trades_lookup[resource_tuple])

	def check_discard_trades(self, player, game_index):
		resource_tuple = tuple(np.minimum(player.resource_cards[game_index], 4))
		np.copyto(player.dynamic_mask.mask_slices[df.discard][game_index], discard_trades_lookup[resource_tuple])

	def check_player_purchase(self, player, game_index):
		self.check_place_settlement(player, game_index)
		self.check_place_city(player, game_index)
		self.check_place_road(player, game_index)
		self.check_buy_development_card(player, game_index)

	def check_place_road(self, player, game_index):
		if player.can_afford(gs.road_cost, game_index):
			player.dynamic_mask.place_road[game_index].fill(True)
			player.apply_static(df.place_road, game_index)

	def check_place_settlement(self, player, game_index):
		if player.can_afford(gs.settlement_cost, game_index) and player.settlement_count[game_index] < gs.max_settlement_count:
			player.dynamic_mask.place_settlement[game_index].fill(True)
			player.apply_static(df.place_settlement, game_index)

	def check_place_city(self, player, game_index):
		if player.can_afford(gs.city_cost, game_index):
			player.dynamic_mask.place_city[game_index].fill(True)
			player.apply_static(df.place_city, game_index)

	def check_buy_development_card(self, player, game_index):
		if player.can_afford(gs.development_card_cost, game_index) and len(self.game.development_card_stack[game_index]) > 0:
			player.dynamic_mask.buy_development_card[game_index].fill(True)
			player.apply_static(df.buy_development_card, game_index)

	def check_player_play_development_card(self, player, game_index):
		if self.game.state.played_development_card_count[game_index] < gs.max_development_cards_played_per_turn:
			if player.development_cards[game_index][gs.knight_index] > 0:
				player.dynamic_mask.play_knight[game_index].fill(True)
			if player.development_cards[game_index][gs.monopoly_index] > 0:
				player.dynamic_mask.play_monopoly[game_index].fill(True)
			if player.development_cards[game_index][gs.road_building_index] > 0:
				player.dynamic_mask.play_road_building[game_index].fill(True)
			if player.development_cards[game_index][gs.year_of_plenty_index] > 0:
				player.dynamic_mask.play_year_of_plenty[game_index].fill(True)

	def check_bank_can_disburse(self, trade, game_index):
		single_recipient = np.where(np.sum(np.where(trade, 1, 0), axis=0) == 1, True, False)
		bank_can_afford = self.check_bank_can_afford(trade, game_index)
		return np.minimum(np.where(np.logical_or(single_recipient, bank_can_afford), trade, 0), self.game.state.bank_resources[game_index])

	def check_bank_can_afford(self, trade, game_index):
		return np.all(self.game.state.bank_resources[game_index] + np.expand_dims(np.sum(trade, axis=0) * -1, axis=0) >= 0, axis=0)

	def set_post_roll_mask(self, player, game_index):
		player.dynamic_mask.only(df.end_turn, game_index)
		self.check_bank_trades(player, game_index)
		self.check_general_port_trades(player, game_index)
		self.check_resource_port_trades(player, game_index)
		self.check_player_trades(player, game_index)
		self.check_player_purchase(player, game_index)
		self.check_player_play_development_card(player, game_index)

	def compare_mask(self, mask):
		for act, mask_value, i in zip(self.action_lookup, mask, range(len(self.action_lookup))):
			callback, args = act
			print(i, callback.__name__, args, mask_value)

	def handle_action(self, action_step, player, game_index):
		action = action_step.action.numpy()[0]
		callback, args = self.action_lookup[action]
		callback(args, player, game_index)
		# player.action_count.append(action)

	# def log_actions(self, callback, args, player):
	# 	log_string = ""
	# 	log_string += str(self.game.num_step).ljust(6)
	# 	log_string += str(np.sum(player.dynamic_mask.mask_slices[df.place_settlement])).ljust(3)
	# 	log_string += str(np.sum(player.dynamic_mask.mask_slices[df.place_road])).ljust(3)
	# 	log_string += str(np.sum(player.dynamic_mask.mask_slices[df.place_city])).ljust(3)
	# 	log_string += str(np.sum(player.dynamic_mask.mask_slices[df.buy_development_card])).ljust(3)
	# 	log_string += str(np.sum(player.dynamic_mask.mask_slices[df.bank_trade])).ljust(3)
	# 	log_string += str(np.sum(player.dynamic_mask.mask_slices[df.general_port_trade])).ljust(3)
	# 	log_string += str(np.sum(player.dynamic_mask.mask_slices[df.resource_port_trade])).ljust(3)
	# 	log_string += str(np.sum(player.dynamic_mask.mask_slices[df.offer_player_trade])).ljust(3)
	# 	log_string += str(np.sum(player.dynamic_mask.mask)).ljust(4)
	# 	log_string += str(player).ljust(95)
	# 	log_string += callback.__name__.ljust(35)[7:]
	# 	log_string += str(args).ljust(20)
	# 	return log_string

	def handle_end_turn(self, _, player, game_index):
		# Reset the current player to non-active status
		player.current_player[game_index].fill(False)
		player.development_cards[game_index] += player.development_card_bought_this_turn[game_index]
		player.development_card_count[game_index] += np.sum(player.development_card_bought_this_turn[game_index])
		player.development_card_bought_this_turn[game_index].fill(0)
		player.dynamic_mask.only(df.no_action, game_index)

		# Reset helper fields
		self.game.player_trades_this_turn[game_index] = 0
		self.game.state.turn_number[game_index] += 1
		self.game.state.bought_development_card_count[game_index].fill(0)
		self.game.state.played_development_card_count[game_index].fill(0)

		# Set next player to be current player
		if self.game.player_order_build_phase[game_index]:
			next_player_index = self.game.player_order_build_phase[game_index].pop(0)
		elif self.game.player_order_build_phase_reversed[game_index]:
			next_player_index = self.game.player_order_build_phase_reversed[game_index].pop(0)
		else:
			if self.game.state.build_phase[game_index]:
				# print("build phase over")
				self.game.state.build_phase[game_index].fill(False)
				for player in self.game.player_list:
					player.static_mask.can(df.buy_development_card, game_index)
			next_player_index = next(self.game.player_cycle[game_index])
		next_player = self.game.player_list[next_player_index]
		# self.game.state.current_player_index[game_index].fill(next_player.index)
		self.game.current_player[game_index] = next_player
		next_player.current_player[game_index].fill(True)
		if self.game.state.build_phase[game_index]:
			next_player.dynamic_mask.only(df.place_settlement, game_index)
			np.logical_and(
				next_player.dynamic_mask.place_settlement[game_index],
				self.game.state.vertex_open[game_index],
				out=next_player.dynamic_mask.place_settlement[game_index])
		else:
			next_player.dynamic_mask.only(df.roll_dice, game_index)
		self.check_player_play_development_card(next_player, game_index)

	def handle_roll_dice(self, roll, player, game_index):
		roll = roll or self.game.dice.roll()
		self.game.state.current_roll[game_index].fill(0)
		self.game.state.current_roll[game_index][roll] = 1
		if roll == gs.robber_activation_roll:
			self.handle_robber_roll(player, game_index)
			return
		self.handle_distribute_resources(roll, player, game_index)
		self.set_post_roll_mask(player, game_index)

	def handle_distribute_resources(self, roll, _, game_index):
		trade = np.zeros((gs.player_count, gs.resource_type_count), dtype=np.int32)
		for tile in self.game.board.roll_hash[game_index][roll]:
			for vertex in tile.vertices:
				if vertex.owned_by[game_index]:
					payout = 1 if vertex.settlement[game_index] else 2 if vertex.city[game_index] else 0
					player_index = vertex.owned_by[game_index].index
					resource = tile.resource[game_index]
					trade[player_index] += payout * resource[:5]
		trade = self.check_bank_can_disburse(trade, game_index)
		for disburse_player in self.game.player_list:
			disburse_player.distribution_total += trade[disburse_player.index]
			self.handle_bank_trade(trade[disburse_player.index], disburse_player, game_index)

	def handle_robber_roll(self, player, game_index):
		some_player_discards = False
		for robbed_player in self.game.player_list:
			if robbed_player.resource_card_count[game_index] > gs.rob_player_above_card_count:
				some_player_discards = True
				robbed_player.dynamic_mask.only(df.discard, game_index)
				self.check_discard_trades(robbed_player, game_index)
				robbed_player.must_discard[game_index].fill(robbed_player.resource_card_count[game_index].item() // 2)
			else:
				robbed_player.dynamic_mask.only(df.no_action, game_index)
		if not some_player_discards:
			player.dynamic_mask.only(df.move_robber, game_index)
			player.must_move_robber[game_index].fill(1)

	def handle_player_trade(self, player_from, player_to, trade, game_index):
		change = np.sum(trade)
		player_from.resource_cards[game_index] += trade
		player_from.resource_card_count[game_index] += change
		player_to.resource_cards[game_index] -= trade
		player_to.resource_card_count[game_index] -= change
		if not self.game.state.build_phase[game_index]:
			self.set_post_roll_mask(self.game.current_player[game_index], game_index)
		assert not np.any(player_from.resource_cards[game_index] < 0), str(player_from.resource_cards[game_index])
		assert not np.any(player_to.resource_cards[game_index] < 0), str(player_to.resource_cards[game_index])

	def handle_bank_trade(self, trade, player, game_index):
		player.resource_cards[game_index] += trade
		self.game.state.bank_resources[game_index] -= trade
		player.resource_card_count[game_index].fill(int(np.sum(player.resource_cards[game_index])))
		assert not np.any(player.resource_cards[game_index] < 0), str(player.resource_cards[game_index]) + str(trade) + player.for_game(game_index)
		if self.game.state.build_phase[game_index]: return
		if player is not self.game.current_player[game_index]: return
		if player.must_discard[game_index] != 0: return
		self.set_post_roll_mask(player, game_index)

	def handle_offer_player_trade(self, trade, player, game_index):
		self.game.player_trades_this_turn[game_index] += 1
		player.dynamic_mask.only(df.no_action, game_index)
		player.offering_trade[game_index].fill(True)
		for opponent in player.other_players:
			opponent.dynamic_mask.only(df.decline_player_trade, game_index)
			if opponent.can_afford(trade * -1, game_index):
				opponent.dynamic_mask.can(df.accept_player_trade, game_index)
		self.game.state.current_player_trade[game_index] += trade
		self.game.trading_player[game_index] = player

	def handle_accept_player_trade(self, _, player, game_index):
		player.accepted_trade[game_index].fill(1)
		player.dynamic_mask.only(df.no_action, game_index)
		self.game.trading_player[game_index].static_mask.confirm_player_trade[game_index][player.index] = True
		self.check_trade_responses(game_index)

	def handle_decline_player_trade(self, _, player, game_index):
		player.declined_trade[game_index].fill(1)
		player.dynamic_mask.only(df.no_action, game_index)
		self.check_trade_responses(game_index)

	def check_trade_responses(self, game_index):
		waiting_on_trade_responses = False
		for opponent in self.game.trading_player[game_index].other_players:
			if not (opponent.accepted_trade[game_index] or opponent.declined_trade[game_index]):
				waiting_on_trade_responses = True
		if not waiting_on_trade_responses:
			self.game.trading_player[game_index].dynamic_mask.only(df.confirm_player_trade, game_index)
			self.game.trading_player[game_index].dynamic_mask.can(df.cancel_player_trade, game_index)
			self.game.trading_player[game_index].apply_static(df.confirm_player_trade, game_index)

	def handle_cancel_player_trade(self, _, player, game_index):
		player.static_mask.confirm_player_trade[game_index].fill(0)
		self.game.trading_player[game_index] = None
		player.offering_trade[game_index].fill(0)
		for opponent in player.other_players:
			opponent.accepted_trade[game_index].fill(0)
			opponent.declined_trade[game_index].fill(0)
		self.game.state.current_player_trade[game_index].fill(0)
		self.set_post_roll_mask(player, game_index)

	def handle_confirm_player_trade(self, trade_player, player, game_index):
		self.handle_player_trade(player, trade_player, self.game.state.current_player_trade[game_index], game_index)
		trade_player.player_trade_total += self.game.state.current_player_trade[game_index]
		player.player_trade_total -= self.game.state.current_player_trade[game_index]
		self.handle_cancel_player_trade(None, player, game_index)

	def handle_move_robber(self, tile, player, game_index):
		player.must_move_robber[game_index].fill(0)
		self.game.board.robbed_tile[game_index].has_robber.fill(False)
		self.game.board.robbed_tile[game_index] = tile
		self.game.board.robbed_tile[game_index].has_robber.fill(True)
		player.dynamic_mask.only(df.rob_player, game_index)
		players_to_rob = 0
		for rob_player in player.other_players:
			if tile.players_to_rob[game_index, player.index, rob_player.index] and np.sum(rob_player.resource_cards[game_index]) > 0:
				player.dynamic_mask.rob_player[game_index][rob_player.index] = True
				players_to_rob += 1
			else:
				player.dynamic_mask.rob_player[game_index][rob_player.index] = False
		player.dynamic_mask.rob_player[game_index][player.index] = False
		if np.sum(player.dynamic_mask.rob_player[game_index]) > 0:
			for opponent in player.other_players:
				opponent.dynamic_mask.only(df.no_action, game_index)
		else:
			self.set_post_roll_mask(player, game_index)

	def handle_rob_player(self, rob_player, player, game_index):
		rob_player_deck = reverse_histogram(rob_player.resource_cards[game_index])
		random.shuffle(rob_player_deck)
		trade = np.zeros(gs.resource_type_count, dtype=np.int32)
		trade[rob_player_deck[0]] = -1 * gs.robber_steals_card_quantity
		self.handle_player_trade(rob_player, player, trade, game_index)
		self.set_post_roll_mask(player, game_index)

	def handle_discard(self, trade, player, game_index):
		self.handle_bank_trade(trade, player, game_index)
		player.discard_total[game_index] -= trade
		player.must_discard[game_index] -= 1
		someone_must_discard = False
		for discard_player in self.game.player_list:
			if discard_player.must_discard[game_index] > 0:
				someone_must_discard = True
				self.check_discard_trades(discard_player, game_index)
			else:
				discard_player.dynamic_mask.only(df.no_action, game_index)
		if not someone_must_discard:
			self.game.current_player[game_index].dynamic_mask.only(df.move_robber, game_index)
			self.game.current_player[game_index].must_move_robber[game_index].fill(1)

	def handle_place_city(self, vertex, player, game_index):
		assert player.city_count[game_index] < gs.max_city_count, str(player.resource_cards[game_index]) + " " + str(player.development_cards[game_index])
		self.handle_bank_trade(gs.city_cost, player, game_index)
		self.give_player_city(vertex, player, game_index)
		self.block_vertex_city(vertex, game_index)
		self.check_max_cities(player, game_index)
		self.set_post_roll_mask(player, game_index)

	def give_player_city(self, vertex, player, game_index):
		player.city_count[game_index] += 1
		player.settlement_count[game_index] -= 1
		player.change_victory_points(gs.city_victory_points - gs.settlement_victory_points, game_index)
		vertex.city[game_index].fill(1)
		vertex.settlement[game_index].fill(0)

	def check_max_cities(self, player, game_index):
		if player.city_count[game_index] == gs.max_city_count:
			player.dynamic_mask.cannot(df.place_city, game_index)
			player.static_mask.cannot(df.place_city, game_index)

	def block_vertex_city(self, vertex, game_index):
		for block_player in self.game.player_list:
			block_player.static_mask.place_city[game_index][vertex.index] = False

	def handle_place_road(self, edge, player, game_index):
		assert player.road_count[game_index] < gs.max_road_count, str(player.resource_cards[game_index]) + " " + str(player.development_cards[game_index])
		for block_player in self.game.player_list:
			block_player.static_mask.place_road[game_index][edge.index] = False
		player.road_count[game_index] += 1
		edge.open[game_index].fill(0)
		player.owned_edges[game_index][edge.index] = True
		player.edge_list[game_index].append(edge)
		if self.game.state.build_phase[game_index]:
			self.game.state.build_phase_placed_road[game_index].fill(1)
			player.dynamic_mask.only(df.end_turn, game_index)
		elif self.game.resolve_road_building_count[game_index] > 0:
			self.game.resolve_road_building_count[game_index] -= 1
			if self.game.resolve_road_building_count[game_index] == 0:
				self.set_post_roll_mask(player, game_index)
		else:
			self.handle_bank_trade(gs.road_cost, player, game_index)
			self.set_post_roll_mask(player, game_index)
		for adjacent_edge in edge.edges:
			player.static_mask.place_road[game_index][adjacent_edge.index] = adjacent_edge.open[game_index]
		for adjacent_vertex in edge.vertices:
			player.static_mask.place_settlement[game_index][adjacent_vertex.index] = adjacent_vertex.open[game_index]
			if adjacent_vertex not in player.edge_proximity_vertices[game_index]:
				player.edge_proximity_vertices[game_index].append(adjacent_vertex)
		if player.road_count[game_index] == gs.max_road_count:
			player.static_mask.place_road[game_index] = False
			player.dynamic_mask.place_road[game_index] = False
		self.maintain_longest_road(player, game_index)

	def check_max_roads(self, player, game_index):
		if player.road_count[game_index] == gs.max_road_count:
			print("Tried to place too many roads", player)
			player.dynamic_mask.cannot(df.place_road, game_index)
			player.static_mask.cannot(df.place_road, game_index)
			return False
		return True

	def maintain_longest_road(self, player, game_index):
		player.calculate_longest_road(game_index)
		if player.longest_road[game_index] > self.game.state.longest_road_length[game_index]:
			np.copyto(self.game.state.longest_road_length[game_index], player.longest_road[game_index])
			if not player.owns_longest_road[game_index]:
				if player.longest_road[game_index] >= gs.min_longest_road:
					if self.game.longest_road_owner[game_index]:
						self.game.longest_road_owner[game_index].set_longest_road(False, game_index)
					player.set_longest_road(True, game_index)

	def handle_place_settlement(self, vertex, player, game_index):
		assert player.settlement_count[game_index] < gs.max_settlement_count, str(player.resource_cards[game_index]) + " " + str(player.development_cards[game_index])
		if self.game.state.build_phase[game_index]:
			self.handle_build_phase_settlement(vertex, player, game_index)
		self.give_player_settlement(vertex, player, game_index)
		self.block_vertex_settlement(vertex, player, game_index)
		self.allow_player_tile_rob(vertex.tiles, player, game_index)
		if not self.game.state.build_phase[game_index]:
			self.handle_bank_trade(gs.settlement_cost, player, game_index)
			self.set_post_roll_mask(player, game_index)

	def check_max_settlement(self, player, game_index):
		if player.settlement_count[game_index] == gs.max_settlement_count:
			print("Tried to place too many settlements", player)
			player.dynamic_mask.cannot(df.place_settlement, game_index)
			return False
		return True

	def handle_build_phase_settlement(self, vertex, player, game_index):
		player.dynamic_mask.cannot(df.place_settlement, game_index)
		self.allow_edge_list(vertex.edges, player, game_index)
		self.game.state.build_phase_placed_settlement[game_index] = 1
		if self.game.state.build_phase_reversed[game_index]:
			self.handle_settlement_disbursement(vertex, player, game_index)

	def handle_settlement_disbursement(self, vertex, player, game_index):
		trade = np.zeros(gs.resource_type_count_tile, dtype=np.int32)
		for tile in vertex.tiles:
			trade[tile.resource_index[game_index]] += 1
		player.starting_distribution[game_index] += trade[:5]
		self.handle_bank_trade(trade[:5], player, game_index)

	def allow_edge_list(self, edge_list, player, game_index):
		for edge in edge_list:
			player.dynamic_mask.mask_slices[df.place_road][game_index][edge.index] = True
			player.static_mask.mask_slices[df.place_road][game_index][edge.index] = True

	def give_player_settlement(self, vertex, player, game_index):
		player.settlement_count[game_index] += 1
		player.change_victory_points(gs.settlement_victory_points, game_index)
		np.logical_or(
			player.port_access[game_index],
			vertex.port[game_index],
			out=player.port_access[game_index])
		vertex.owned_by[game_index] = player
		vertex.settlement[game_index].fill(True)
		player.owned_vertices[game_index][vertex.index] = True

	def block_vertex_settlement(self, vertex, player, game_index):
		for vertex in [vertex, *vertex.vertices]:
			vertex.open[game_index].fill(False)
			for block_player in self.game.player_list:
				block_player.static_mask.place_settlement[game_index][vertex.index] = False
				block_player.static_mask.place_city[game_index][vertex.index] = False
		player.static_mask.place_city[game_index][vertex.index] = True

	def allow_player_tile_rob(self, tile_list, player, game_index):
		for adjacent_tile in tile_list:
			adjacent_tile.players_to_rob[game_index, :, player.index] = True
			adjacent_tile.players_to_rob[game_index, player.index, player.index] = False

	def handle_buy_development_card(self, _, player, game_index):
		assert len(self.game.development_card_stack[game_index]) > 0, str(player.resource_cards[game_index]) + " " + str(player.development_cards[game_index])
		if not self.check_no_development_cards(player, game_index):
			return
		self.handle_bank_trade(gs.development_card_cost, player, game_index)
		card_index = self.game.development_card_stack[game_index].pop(0)
		self.game.state.bought_development_card_count[game_index] += 1
		self.game.state.bank_development_card_count[game_index] -= 1
		if len(self.game.development_card_stack[game_index]) == 0:
			for player in self.game.player_list:
				player.static_mask.cannot(df.buy_development_card, game_index)
		if card_index == gs.victory_point_card_index:
			player.development_cards[game_index][card_index] += 1
			player.check_victory(game_index)
		else:
			player.development_card_bought_this_turn[game_index][card_index] += 1
		self.set_post_roll_mask(player, game_index)

	def check_no_development_cards(self, player, game_index):
		if len(self.game.development_card_stack[game_index]) == 0:
			print("Tried to buy too many dev cards", player)
			player.dynamic_mask.cannot(df.buy_development_card, game_index)
			return False
		return True

	def handle_play_dev_card(self, card_index, player, game_index):
		player.development_cards[game_index][card_index] -= 1
		player.development_cards_played[game_index][card_index] += 1
		player.development_card_count[game_index] -= 1
		self.game.state.played_development_card_count[game_index] += 1
		self.set_post_roll_mask(player, game_index)

	def handle_play_knight(self, _, player, game_index):
		self.handle_play_dev_card(gs.knight_index, player, game_index)
		player.dynamic_mask.only(df.move_robber, game_index)
		player.must_move_robber[game_index].fill(1)
		self.maintain_largest_army(player, game_index)

	def maintain_largest_army(self, player, game_index):
		if player.development_cards_played[game_index][gs.knight_index] >= gs.min_largest_army:
			if player.development_cards_played[game_index][gs.knight_index] > self.game.state.largest_army_size[game_index]:
				if not player.owns_largest_army[game_index]:
					if self.game.largest_army_owner[game_index]:
						self.game.largest_army_owner[game_index].largest_army(False, game_index)
					player.largest_army(True, game_index)
				else:
					np.copyto(
						self.game.state.largest_army_size[game_index],
						player.development_cards_played[game_index][gs.knight_index])

	def handle_play_monopoly(self, resource_index, player, game_index):
		self.handle_play_dev_card(gs.monopoly_index, player, game_index)
		for opponent in player.other_players:
			trade = np.zeros(gs.resource_type_count, dtype=np.int32)
			count = opponent.resource_cards[game_index][resource_index]
			trade[resource_index] = -1 * count
			self.handle_player_trade(opponent, player, trade, game_index)

	def handle_play_road_building(self, _, player, game_index):
		self.handle_play_dev_card(gs.road_building_index, player, game_index)
		roads_to_build = min(gs.road_building_road_count, player.road_count[game_index] - gs.max_road_count)
		if roads_to_build > 0:
			self.game.resolve_road_building_count[game_index] = roads_to_build
			player.dynamic_mask.only(df.place_road, game_index)
			player.apply_static(df.place_road, game_index)

	def handle_play_year_of_plenty(self, trade, player, game_index):
		self.handle_play_dev_card(gs.year_of_plenty_index, player, game_index)
		self.handle_bank_trade(trade, player, game_index)

	def handle_no_action(self, _, __, ___):
		pass
