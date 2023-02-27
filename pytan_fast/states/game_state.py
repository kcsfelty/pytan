from pytan_fast.definitions import *
import pytan_fast.settings as cs


game_state_degrees = {
	bank_development_card_count: 1,
	longest_road_length: 1,
	largest_army_size: 1,
	turn_number: 1,
	build_phase: 1,
	build_phase_reversed: 1,
	build_phase_placed_settlement: 1,
	build_phase_placed_road: 1,
	bought_development_card_count: 1,
	played_development_card_count: 1,
	vertex_settlement: cs.vertex_count,
	vertex_city: cs.vertex_count,
	vertex_open: cs.vertex_count,
	edge_open: cs.edge_count,
	tile_has_robber: cs.tile_count,
	tile_resource: cs.tile_count * cs.resource_type_count_tile,
	tile_roll_number: cs.tile_count * cs.tile_roll_number_type_count,
	bank_resources: cs.resource_type_count,
	current_player_trade: cs.resource_type_count,
	current_roll: cs.tile_roll_number_type_count,
	vertex_has_port: cs.vertex_count * cs.port_type_count,
}
