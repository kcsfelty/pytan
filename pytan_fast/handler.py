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
	def __init__(self, game, log_action=False):
		self.game = game
		self.log_action = log_action
		self.bank_trades, self.general_port_trades, self.resource_port_trades, self.player_trades, self.year_of_plenty_trades, self.discard_trades = get_trades()
		self.bank_trades = np.array(self.bank_trades)
		self.general_port_trades = np.array(self.general_port_trades)
		self.resource_port_trades = np.array(self.resource_port_trades)
		self.player_trades = np.array(self.player_trades)
		self.year_of_plenty_trades = np.array(self.year_of_plenty_trades)

		self.bank_trades_lookup_bank = get_trade_lookup(self.bank_trades * -1)
		self.general_port_trades_lookup_bank = get_trade_lookup(self.general_port_trades * -1)
		self.resource_port_trades_lookup_bank = get_trade_lookup(self.resource_port_trades * -1)
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
		player_can_afford = self.bank_trades_lookup[resource_tuple]
		bank_resource_tuple = tuple(np.minimum(self.game.state.game_state_slices[df.bank_resources], 1))
		bank_can_afford = self.bank_trades_lookup_bank[bank_resource_tuple]
		np.copyto(player.dynamic_mask.mask_slices[df.bank_trade], np.logical_and(player_can_afford, bank_can_afford))
		# self.check_general_port_trades(player)
		# resource_tuple = tuple(np.minimum(player.resource_cards, 4) * player.port_access[:5])
		# np.copyto(player.dynamic_mask.resource_port_trade, self.resource_port_trades_lookup[resource_tuple])

	def check_general_port_trades(self, player):
		if player.port_access[5]:
			resource_tuple = tuple(np.minimum(player.resource_cards, 4))
			player_can_afford = self.bank_trades_lookup[resource_tuple]
			bank_resource_tuple = tuple(np.minimum(self.game.state.game_state_slices[df.bank_resources], 1))
			bank_can_afford = self.general_port_trades_lookup_bank[bank_resource_tuple]
			np.copyto(player.dynamic_mask.mask_slices[df.general_port_trade], np.logical_and(player_can_afford, bank_can_afford))

	def check_resource_port_trades(self, player):
		resource_tuple = tuple(np.minimum(player.resource_cards, 4) * player.port_access[:5])
		player_can_afford = self.bank_trades_lookup[resource_tuple]
		bank_resource_tuple = tuple(np.minimum(self.game.state.game_state_slices[df.bank_resources], 1))
		bank_can_afford = self.resource_port_trades_lookup_bank[bank_resource_tuple]
		np.copyto(player.dynamic_mask.mask_slices[df.resource_port_trade], np.logical_and(player_can_afford, bank_can_afford))

	def check_player_trades(self, player):
		if self.game.player_trades_this_turn is not gs.player_trade_per_turn_limit:
			resource_tuple = tuple(np.minimum(player.resource_cards, 4))
			np.copyto(player.dynamic_mask.mask_slices[df.offer_player_trade], self.player_trades_lookup[resource_tuple])

	def check_discard_trades(self, player):
		resource_tuple = tuple(np.minimum(player.resource_cards, 4))
		np.copyto(player.dynamic_mask.mask_slices[df.discard], self.discard_trades_lookup[resource_tuple])

	def check_player_purchase(self, player):
		self.check_place_road(player)
		self.check_place_settlement(player)
		self.check_place_city(player)
		self.check_buy_development_card(player)

	def check_place_road(self, player):
		# if player.can_afford(gs.road_cost) and player.road_count < gs.max_road_count:
		player.dynamic_mask.mask_slices[df.place_road].fill(player.can_afford(gs.road_cost))
		player.apply_static(df.place_road)

	def check_place_settlement(self, player):
		if player.can_afford(gs.settlement_cost) and player.settlement_count < gs.max_settlement_count:
			player.dynamic_mask.mask_slices[df.place_settlement].fill(True)
			player.apply_static(df.place_settlement)

	def check_place_city(self, player):
		if player.can_afford(gs.city_cost):
			player.dynamic_mask.mask_slices[df.place_city].fill(True)
			player.apply_static(df.place_city)

	def check_buy_development_card(self, player):
		if player.can_afford(gs.development_card_cost) and len(self.game.development_card_stack) > 0:
			player.dynamic_mask.mask_slices[df.buy_development_card].fill(True)
			player.apply_static(df.buy_development_card)

	def check_player_play_development_card(self, player):
		if self.game.state.game_state_slices[df.played_development_card_count] is not gs.max_development_cards_played_per_turn:
			player.dynamic_mask.mask_slices[df.play_knight].fill(player.development_cards[gs.knight_index] > 0)
			player.dynamic_mask.mask_slices[df.play_monopoly].fill(player.development_cards[gs.monopoly_index] > 0)
			player.dynamic_mask.mask_slices[df.play_road_building].fill(player.development_cards[gs.road_building_index] > 0)
			player.dynamic_mask.mask_slices[df.play_year_of_plenty].fill(player.development_cards[gs.year_of_plenty_index] > 0)

	def check_bank_can_disburse(self, trade):
		single_recipient = np.where(np.sum(np.where(trade, 1, 0), axis=0) == 1, True, False)
		bank_can_afford = self.check_bank_can_afford(trade)
		return np.minimum(np.where(np.logical_or(single_recipient, bank_can_afford), trade, 0), self.game.state.game_state_slices[df.bank_resources])

	def check_bank_can_afford(self, trade):
		return np.all(self.game.state.game_state_slices[df.bank_resources] + np.expand_dims(np.sum(trade, axis=0) * -1, axis=0) >= 0, axis=0)

	def set_post_roll_mask(self, player):
		player.dynamic_mask.only(df.end_turn)
		self.check_bank_trades(player)
		self.check_general_port_trades(player)
		self.check_resource_port_trades(player)
		self.check_player_trades(player)
		self.check_player_purchase(player)
		self.check_player_play_development_card(player)

	def compare_mask(self, mask):
		for act, mask_value, i in zip(self.action_lookup, mask, range(len(self.action_lookup))):
			callback, args = act
			print(i, callback.__name__, args, mask_value)

	def handle_action(self, action_step, player):
		action = action_step.action.numpy()[0]
		callback, args = self.action_lookup[action]
		if self.log_action:
			print(self.log_actions(action, callback, args, player))
		player.action_count.append(action)
		callback(args, player)

		assert player.can_afford(no_cost), self.log_actions(action, callback, args, player)
		assert player.road_count <= 15, self.log_actions(action, callback, args, player)
		assert player.road_count >= 0, self.log_actions(action, callback, args, player)
		assert player.settlement_count <= 5, self.log_actions(action, callback, args, player)
		assert player.settlement_count >= 0, self.log_actions(action, callback, args, player)
		assert player.city_count <= 4, self.log_actions(action, callback, args, player)
		assert player.city_count >= 0, self.log_actions(action, callback, args, player)
		assert player.development_card_count >= 0, self.log_actions(action, callback, args, player)

	def log_actions(self, action, callback, args, player):
		log_string = ""
		log_string += str(self.game.num_step).ljust(6)
		log_string += str(np.sum(player.dynamic_mask.mask_slices[df.place_settlement])).ljust(3)
		log_string += str(np.sum(player.dynamic_mask.mask_slices[df.place_road])).ljust(3)
		log_string += str(np.sum(player.dynamic_mask.mask_slices[df.place_city])).ljust(3)
		log_string += str(np.sum(player.dynamic_mask.mask_slices[df.buy_development_card])).ljust(3)
		log_string += str(np.sum(player.dynamic_mask.mask_slices[df.bank_trade])).ljust(3)
		log_string += str(np.sum(player.dynamic_mask.mask_slices[df.general_port_trade])).ljust(3)
		log_string += str(np.sum(player.dynamic_mask.mask_slices[df.resource_port_trade])).ljust(3)
		log_string += str(np.sum(player.dynamic_mask.mask_slices[df.offer_player_trade])).ljust(3)
		log_string += str(np.sum(player.dynamic_mask.mask)).ljust(4)
		log_string += str(player).ljust(95)
		log_string += callback.__name__.ljust(35)[7:]
		log_string += str(args).ljust(20)
		return log_string

	def handle_end_turn(self, _, player):
		next_player = next(self.game.player_cycle)
		self.game.current_player = next_player
		player.current_player.fill(0)
		player.development_cards += player.development_card_bought_this_turn
		player.development_card_count += np.sum(player.development_card_bought_this_turn)
		player.development_card_bought_this_turn.fill(0)
		player.dynamic_mask.only(df.no_action)
		self.game.player_trades_this_turn = 0
		self.game.state.game_state_slices[df.turn_number] += 1
		self.game.current_player = next_player
		next_player.current_player.fill(1)
		next_player.dynamic_mask.only(df.roll_dice)
		self.check_player_play_development_card(next_player)
		self.game.state.game_state_slices[df.bought_development_card_count].fill(0)
		self.game.state.game_state_slices[df.played_development_card_count].fill(0)

	def handle_roll_dice(self, roll, player):
		roll = roll or self.game.dice.roll()
		self.game.state.game_state_slices[df.current_roll].fill(0)
		self.game.state.game_state_slices[df.current_roll][roll].fill(1)
		if roll == gs.robber_activation_roll:
			self.handle_robber_roll(player)
			return
		self.handle_distribute_resources(roll, player)
		self.set_post_roll_mask(player)

	def handle_distribute_resources(self, roll, _):
		trade = np.zeros((gs.player_count, gs.resource_type_count), dtype=np.int32)
		for tile in self.game.board.roll_hash[roll]:
			for vertex in tile.vertices:
				if vertex.owned_by:
					trade[vertex.owned_by.index][tile.resource_index] += 1 if vertex.settlement else 2 if vertex.city else 0
		trade = self.check_bank_can_disburse(trade)
		for disburse_player in self.game.player_list:
			disburse_player.distribution_total += trade[disburse_player.index]
			self.handle_bank_trade(trade[disburse_player.index], disburse_player)

	def handle_robber_roll(self, player):
		some_player_discards = False
		for robbed_player in self.game.player_list:
			if robbed_player.resource_card_count > gs.rob_player_above_card_count:
				some_player_discards = True
				robbed_player.dynamic_mask.mask.fill(False)
				robbed_player.dynamic_mask.only(df.discard)
				self.check_discard_trades(robbed_player)
				robbed_player.must_discard = robbed_player.resource_card_count.item() // 2
				for _ in range(robbed_player.must_discard):
					self.game.immediate_play.insert(0, robbed_player)
			else:
				robbed_player.dynamic_mask.only(df.no_action)
		if not some_player_discards:
			player.dynamic_mask.only(df.move_robber)
			self.game.immediate_play.append(player)

	def handle_player_trade(self, player_from, player_to, trade):
		change = np.sum(trade)
		player_from.resource_cards += trade
		player_from.resource_card_count += change
		player_to.resource_cards -= trade
		player_to.resource_card_count -= change
		assert player_from.can_afford(no_cost), (player_from, trade)
		assert player_to.can_afford(no_cost), (player_to, trade)
		if not self.game.state.game_state_slices[df.build_phase]:
			self.set_post_roll_mask(self.game.current_player)

	def handle_bank_trade(self, trade, player):
		player.resource_cards += trade
		self.game.state.game_state_slices[df.bank_resources] -= trade
		player.resource_card_count = np.sum(player.resource_cards)
		assert player.can_afford(no_cost), (player, trade)
		if not self.game.state.game_state_slices[df.build_phase]:
			if player.must_discard == 0:
				if player == self.game.current_player:
					self.set_post_roll_mask(player)

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
		self.game.trading_player.dynamic_mask.mask_slices[df.confirm_player_trade][player.index:player.index+1].fill(True)

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
		trade_player.player_trade_total += self.game.state.game_state_slices[df.current_player_trade]
		player.player_trade_total -= self.game.state.game_state_slices[df.current_player_trade]
		self.handle_cancel_player_trade(None, player)

	def handle_move_robber(self, tile, player):
		player.must_move_robber.fill(0)
		self.game.board.robbed_tile.has_robber.fill(0)
		self.game.board.robbed_tile = tile
		self.game.board.robbed_tile.has_robber.fill(1)
		player.dynamic_mask.only(df.rob_player)
		players_to_rob = 0
		for rob_player in player.other_players:
			if tile.players_to_rob[player.index, rob_player.index] and np.sum(rob_player.resource_cards) > 0:
				player.dynamic_mask.mask_slices[df.rob_player][rob_player.index:rob_player.index + 1].fill(True)
				players_to_rob += 1
			else:
				player.dynamic_mask.mask_slices[df.rob_player][rob_player.index:rob_player.index + 1].fill(False)
		# for rob_player in self.game.player_list:
		# 	if tile.players_to_rob[player.index, rob_player.index] and np.sum(rob_player.resource_cards) > 0 and rob_player.index is not player.index:
		# 		player.dynamic_mask.mask_slices[df.rob_player][rob_player.index:rob_player.index + 1].fill(True)
		# 		# if np.sum(rob_player.resource_cards) > 0:
		# 		# 	if rob_player.index is not player.index:
		# 		# 		player.dynamic_mask.mask_slices[df.rob_player][rob_player.index:rob_player.index+1].fill(True)
		# 		players_to_rob += 1
		player.dynamic_mask.mask_slices[df.rob_player][player.index:player.index + 1].fill(False)
		if np.sum(player.dynamic_mask.mask_slices[df.rob_player]) > 0:
			self.game.immediate_play.append(player)
		else:
			self.set_post_roll_mask(player)

	def handle_rob_player(self, rob_player, player):
		rob_player_deck = reverse_histogram(rob_player.resource_cards)
		random.shuffle(rob_player_deck)
		trade = np.zeros(gs.resource_type_count, dtype=np.int32)
		trade[rob_player_deck[0]] = -1 * gs.robber_steals_card_quantity
		self.handle_player_trade(rob_player, player, trade)
		rob_player.stolen_total += trade
		player.steal_total += trade
		self.set_post_roll_mask(player)

	def handle_discard(self, trade, player):
		self.handle_bank_trade(trade, player)
		player.discard_total -= trade
		player.must_discard -= 1
		self.check_discard_trades(player)
		if len(self.game.immediate_play) == 0:
			self.game.immediate_play.append(self.game.current_player)
			self.game.current_player.dynamic_mask.only(df.move_robber)

	def handle_place_city(self, vertex, player):
		self.handle_bank_trade(gs.city_cost, player)
		player.city_count += 1
		player.settlement_count -= 1
		player.change_victory_points(gs.city_victory_points - gs.settlement_victory_points)
		vertex.city.fill(1)
		vertex.settlement.fill(0)
		# player.static_mask.place_city[vertex.index].fill(0)
		for block_player in self.game.player_list:
			# block_player.static_mask.place_settlement[vertex.index].fill(0)
			block_player.static_mask.mask_slices[df.place_city][vertex.index:vertex.index+1].fill(False)
		if player.city_count == gs.max_city_count:
			# player.static_mask.mask_slices[df.place_city].fill(False)
			player.static_mask.cannot(df.place_city)
		self.set_post_roll_mask(player)

	def handle_place_road(self, edge, player):
		for block_player in self.game.player_list:
			block_player.static_mask.mask_slices[df.place_road][edge.index:edge.index+1].fill(False)
		player.road_count += 1
		edge.open.fill(0)
		player.owned_edges[edge.index] = 1
		player.edge_list.append(edge)
		if self.game.state.game_state_slices[df.build_phase]:
			self.game.state.game_state_slices[df.build_phase_placed_road] = 1
		elif self.game.resolve_road_building_count > 0:
			self.game.resolve_road_building_count -= 1
			if self.game.resolve_road_building_count == 0:
				self.set_post_roll_mask(player)
		else:
			self.handle_bank_trade(gs.road_cost, player)
		for adjacent_edge in edge.edges:
			if adjacent_edge.open:
				player.static_mask.mask_slices[df.place_road][adjacent_edge.index:adjacent_edge.index+1].fill(True)
		for adjacent_vertex in edge.vertices:
			if adjacent_vertex.open:
				player.static_mask.mask_slices[df.place_settlement][adjacent_vertex.index:adjacent_vertex.index+1].fill(True)
			if adjacent_vertex not in player.edge_proximity_vertices:
				player.edge_proximity_vertices.append(adjacent_vertex)
		if player.road_count == gs.max_road_count:
			player.static_mask.mask_slices[df.place_road].fill(0)
		self.set_post_roll_mask(player)
		self.maintain_longest_road(player)

	def maintain_longest_road(self, player):
		player.calculate_longest_road()
		if player.longest_road > self.game.state.game_state_slices[df.longest_road_length]:
			np.copyto(self.game.state.game_state_slices[df.longest_road_length], player.longest_road)
			if not player.owns_longest_road:
				if player.longest_road >= gs.min_longest_road:
					if self.game.longest_road_owner:
						self.game.longest_road_owner.set_longest_road(False)
					player.set_longest_road(True)

	def handle_place_settlement(self, vertex, player):
		if self.game.state.game_state_slices[df.build_phase]:
			player.dynamic_mask.cannot(df.place_settlement)
			for edge in vertex.edges:
				player.dynamic_mask.mask_slices[df.place_road][edge.index:edge.index+1].fill(True)
				player.static_mask.mask_slices[df.place_road][edge.index:edge.index+1].fill(True)
			self.game.state.game_state_slices[df.build_phase_placed_settlement] = 1
			if self.game.state.game_state_slices[df.build_phase_reversed]:
				trade = np.zeros(gs.resource_type_count_tile, dtype=np.int32)
				for tile in vertex.tiles:
					trade[tile.resource_index] += 1
				player.starting_distribution += trade[:5]
				self.handle_bank_trade(trade[:5], player)
		else:
			self.handle_bank_trade(gs.settlement_cost, player)
		player.settlement_count += 1
		player.change_victory_points(gs.settlement_victory_points)
		np.logical_or(player.port_access, vertex.port, out=player.port_access)
		vertex.owned_by = player
		vertex.settlement.fill(1)
		player.owned_vertices[vertex.index] = 1
		for block_vertex in [vertex, *vertex.vertices]:
			block_vertex.open.fill(0)
			for block_player in self.game.player_list:
				block_player.static_mask.mask_slices[df.place_settlement][block_vertex.index:block_vertex.index+1].fill(False)
				block_player.static_mask.mask_slices[df.place_city][block_vertex.index:block_vertex.index + 1].fill(False)
				# block_player.static_mask.place_settlement[block_vertex.index:block_vertex.index+1] = False
				# block_player.static_mask.place_city[block_vertex.index:block_vertex.index+1] = False
		# player.static_mask.place_city[vertex.index:vertex.index+1] = True
		player.static_mask.mask_slices[df.place_city][vertex.index:vertex.index+1].fill(True)
		for adjacent_tile in vertex.tiles:
			adjacent_tile.players_to_rob[:, player.index] = True
			adjacent_tile.players_to_rob[player.index, player.index] = False
		if not self.game.state.game_state_slices[df.build_phase]:
			self.set_post_roll_mask(player)

	def handle_buy_development_card(self, _, player):
		if len(self.game.development_card_stack) == 0:
			print("tried to draw from empty dev card stack", player)
			return
		self.handle_bank_trade(gs.development_card_cost, player)
		card_index = self.game.development_card_stack.pop(0)
		self.game.state.game_state_slices[df.bought_development_card_count] += 1
		self.game.state.game_state_slices[df.bank_development_card_count] -= 1
		if self.game.state.game_state_slices[df.bank_development_card_count] == 0 or len(self.game.development_card_stack) == 0:
			for player in self.game.player_list:
				player.static_mask.cannot(df.buy_development_card)
				# player.static_mask.mask_slices[df.buy_development_card].fill(False)
		if card_index == gs.victory_point_card_index:
			player.development_cards[card_index] += 1
			player.check_victory()
		else:
			player.development_card_bought_this_turn[card_index] += 1
		self.set_post_roll_mask(player)

	def handle_play_dev_card(self, card_index, player):
		player.development_cards[card_index] -= 1
		player.development_cards_played[card_index] += 1
		player.development_card_count -= 1
		self.game.state.game_state_slices[df.played_development_card_count] += 1
		self.set_post_roll_mask(player)

	def handle_play_knight(self, _, player):
		self.handle_play_dev_card(gs.knight_index, player)
		self.game.immediate_play.append(player)
		player.dynamic_mask.only(df.move_robber)
		player.must_move_robber.fill(1)
		self.maintain_largest_army(player)

	def maintain_largest_army(self, player):
		if player.development_cards_played[gs.knight_index] >= gs.min_largest_army:
			if player.development_cards_played[gs.knight_index] > self.game.state.game_state_slices[df.largest_army_size]:
				if not player.owns_largest_army:
					if self.game.largest_army_owner:
						self.game.largest_army_owner.largest_army(False)
					player.largest_army(True)
				else:
					np.copyto(self.game.state.game_state_slices[df.largest_army_size], player.development_cards_played[gs.knight_index])

	def handle_play_monopoly(self, resource_index, player):
		self.handle_play_dev_card(gs.monopoly_index, player)
		for opponent in player.other_players:
			trade = np.zeros(gs.resource_type_count, dtype=np.int32)
			count = opponent.resource_cards[resource_index]
			trade[resource_index] = -1 * count
			self.handle_player_trade(opponent, player, trade)

	def handle_play_road_building(self, _, player):
		self.handle_play_dev_card(gs.road_building_index, player)
		roads_to_build = min(gs.road_building_road_count, player.road_count - gs.max_road_count)
		if roads_to_build > 0:
			self.game.resolve_road_building_count = roads_to_build
			self.game.immediate_play.extend([player] * roads_to_build)
			player.dynamic_mask.only(df.place_road)
			player.apply_static(df.place_road)

	def handle_play_year_of_plenty(self, trade, player):
		self.handle_play_dev_card(gs.year_of_plenty_index, player)
		self.handle_bank_trade(trade, player)

	def handle_no_action(self, _, player):
		pass
