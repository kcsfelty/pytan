from pytan_fast import settings


def handle_roll_dice(_, roll_player_index, self):
	self.player_rolled = True
	new_roll = self.dice.roll()
	for roll_number in settings.roll_list:
		if roll_number == new_roll:
			self.set_current_roll(roll_number, 1)
		else:
			self.set_current_roll(roll_number, 0)

	if new_roll == 5:
		self.calculate_cards()
		for player_index in settings.player_list:
			card_count = self.get_player_resource_card_count(player_index)
			if card_count > settings.rob_player_above_card_count:
				self.discard_block_state = True
				self.player_discard_count[player_index] = card_count // 2

		self.set_player_must_move_robber(roll_player_index, 1)
		self.move_robber_block_state = True
		return

	total_dispersal_per_resource = [[0 for _ in settings.resource_list] for _ in settings.resource_list]
	player_dispersal_per_resource = [[[0 for _ in settings.resource_list] for _ in settings.player_list] for _ in settings.resource_list]
	for tile in self.roll_tile_hash[new_roll]:
		if self.robber_position is not tile:
			resource_index = self.tile_resource_hash[tile]
			for vertex in self.tile_vertices[tile]:
				gain = 0
				if self.get_vertex_settlement(vertex):
					gain = settings.settlement_distribute_resource_count
				elif self.get_vertex_city(vertex):
					gain = settings.city_distribute_resource_count
				for owner_index in settings.player_list:
					if self.get_vertex_owned(owner_index, vertex):

						player_dispersal_per_resource[resource_index][owner_index][resource_index] += gain
						total_dispersal_per_resource[resource_index][resource_index] += gain
		else:
			# robber blocked dispersal
			pass
	for total_resource_trade, player_resource_trade_list in zip(total_dispersal_per_resource, player_dispersal_per_resource):
		if self.bank_can_afford(total_resource_trade):
			for player_index, player_trade in zip(settings.player_list, player_resource_trade_list):
				if sum(player_trade) > 0:
					self.trade_with_bank(player_index, player_trade)
		else:
			players_disbursed_to = [sum(player_trade) > 0 for player_trade in player_resource_trade_list]
			if players_disbursed_to.count(True) == 1:
				single_player_index = players_disbursed_to.index(True)
				self.trade_with_bank(single_player_index, player_resource_trade_list[single_player_index])

	self.calculate_cards()
