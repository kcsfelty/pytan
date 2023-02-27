def handle_rob_player(rob_player_index, player_index, self):
	self.rob_player(player_index, rob_player_index)
	self.players_to_rob = []
	self.rob_player_block_state = False
	self.rob_player_responsible = -1
