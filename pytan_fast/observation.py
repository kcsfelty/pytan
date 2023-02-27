import numpy as np
from pytan_fast.definitions import *
from pytan_fast.settings import *

game_observation = {
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

	tile_has_robber: tile_count,
	tile_resource: tile_count * resource_type_count_tile,
	tile_roll_number: tile_count * roll_number_type_count,

	vertex_settlement: vertex_count,
	vertex_city: vertex_count,
	vertex_open: vertex_count,

	edge_open: edge_count,

	bank_resources: resource_type_count,
	current_player_trade: resource_type_count,
	current_roll: roll_number_type_count,
	vertex_has_port: vertex_count * port_type_count,
}

public_observation = {
	current_player: player_count,
	must_move_robber: player_count,
	victory_points: player_count,
	resource_card_count: player_count,
	development_card_count: player_count,
	settlement_count: player_count,
	city_count: player_count,
	road_count: player_count,
	longest_road: player_count,
	owns_longest_road: player_count,
	owns_largest_army: player_count,
	offering_trade: player_count,
	accepted_trade: player_count,
	declined_trade: player_count,
	must_discard: player_count,
	vertex_owned: player_count * vertex_count,
	edge_owned: player_count * edge_count,
	port_access: player_count * port_type_count,
	development_cards_played: player_count * development_card_type_count,

}

private_observation = {
	resource_cards: resource_type_count,
	development_cards: development_card_type_count,
	development_card_bought: development_card_type_count,

}

def get_observation():
	observation_term_lengths = {
		# Monoids
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
		# one for each vertex
		vertex_settlement: vertex_count,
		vertex_city: vertex_count,
		vertex_open: vertex_count,
		# one for each edge
		edge_open: edge_count,
		# one for each tile
		tile_has_robber: tile_count,
		# One for each tile for each resource type
		tile_resource: tile_count * resource_type_count_tile,
		# one for each tile, for each roll number
		tile_roll_number: tile_count * roll_number_count,
		# one for each resource
		bank_resources: resource_type_count,
		current_player_trade: resource_type_count,
		# one for each roll number
		current_roll: roll_number_count,
		# one for each vertex for each port type
		vertex_has_port: vertex_count * port_type_count,
		# One per player monoids
		current_player: player_count,
		must_move_robber: player_count,
		victory_points: player_count,
		resource_card_count: player_count,
		development_card_count: player_count,
		settlement_count: player_count,
		city_count: player_count,
		road_count: player_count,
		longest_road: player_count,
		owns_longest_road: player_count,
		owns_largest_army: player_count,
		offering_trade: player_count,
		accepted_trade: player_count,
		declined_trade: player_count,
		must_discard: player_count,
		# One per player per development card type
		development_cards_played: player_count * development_card_type_count,
		# One per player per vertex
		vertex_owned: player_count * vertex_count,
		# One per player per edge
		edge_owned: player_count * edge_count,
		# One per player per port type
		port_access: player_count * port_type_count,

		# Private
		# One per player per resource type
		resource_type_count: resource_type_count,
		# One per player per development card type
		development_type_count: development_card_type_count,
		# One per player per development card type
		development_type_bought_count: development_card_type_count
	}

	observation_length = sum([observation_term_lengths[key] for key in observation_term_lengths])

	observation = np.zeros((player_count, observation_length))
	observation_slices = {}
	current_index = 0
	for term in observation_term_lengths:
		next_index = current_index + observation_term_lengths[term]
		observation_slices[term] = observation[:, current_index:next_index].view()
		current_index = next_index
	# Reshape anything which is one-hot encoded
	observation_slices[tile_resource].shape = (player_count, tile_count, resource_type_count_tile)
	observation_slices[tile_roll_number].shape = (player_count, tile_count, roll_number_count)
	observation_slices[vertex_owned].shape = (player_count, vertex_count, player_count)
	observation_slices[edge_owned].shape = (player_count, edge_count, player_count)
	observation_slices[vertex_has_port].shape = (player_count, vertex_count, port_type_count)
	observation_slices[development_cards_played].shape = (player_count, player_count, development_card_type_count)
	observation_slices[port_access].shape = (player_count, player_count, port_type_count)
	return observation, observation_slices
