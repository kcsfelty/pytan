from pytan_fast import settings

def handle_end_turn(_, player_index, self):
	self.set_turn_number(self.get_turn_number() + 1)
	self.player_rolled = False

	self.current_trades_turn = 0

	if self.get_build_phase():
		self.build_phase_settlement_placed = False
		self.build_phase_road_placed = False
		if len(self.build_phase_order) == 3:
			self.set_build_phase_reversed(1)
		if not self.build_phase_order:
			print("build phase ending")
			self.need_render = True
			next_player_index = 0
			self.set_build_phase(0)
			self.set_build_phase_reversed(0)
		else:
			next_player_index = self.build_phase_order.pop(0)
	else:
		next_player_index = (player_index + 1) % settings.player_count
	for development_card_index in settings.development_card_list:
		add_count = self.get_private_development_card_bought_this_turn_count(player_index, development_card_index)
		self.set_private_development_card_bought_this_turn_count(player_index, development_card_index, 0)
		base_count = self.get_private_development_card_count(player_index, development_card_index)
		self.set_private_development_card_count(player_index, development_card_index, add_count + base_count)
	self.set_player_is_current_player(player_index, 0)
	self.set_player_is_current_player(next_player_index, 1)
