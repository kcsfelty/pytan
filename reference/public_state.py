from reference.definitions import *
import reference.settings as cs


public_state_degrees = {
	current_player: 1,
	must_move_robber: 1,
	must_discard: 1,
	victory_points: 1,
	resource_card_count: 1,
	development_card_count: 1,
	settlement_count: 1,
	city_count: 1,
	road_count: 1,
	longest_road: 1,
	owns_longest_road: 1,
	owns_largest_army: 1,
	offering_trade: 1,
	accepted_trade: 1,
	declined_trade: 1,
	development_cards_played: cs.development_type_count,
	vertex_owned: cs.vertex_count,
	edge_owned: cs.edge_count,
	port_access: cs.port_type_count,
}
