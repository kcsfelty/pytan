from pytan_fast import settings


def handle_play_knight(_, player_index, self):
	self.play_development_card(player_index, settings.knight_index)
	self.set_player_must_move_robber(player_index, 1)
	self.move_robber_block_state = True
	self.calculate_largest_army()
