from pytan_fast.definitions import *
import pytan_fast.settings as cs


action_mapping = {
	end_turn: 1,
	roll_dice: 1,
	bank_trade: cs.resource_type_count * (cs.resource_type_count - 1),
	general_port_trade: cs.resource_type_count * (cs.resource_type_count - 1),
	resource_port_trade: cs.resource_type_count * (cs.resource_type_count - 1),
	offer_player_trade: cs.resource_type_count * 14,
	accept_player_trade: 1,
	decline_player_trade: 1,
	cancel_player_trade: 1,
	confirm_player_trade: cs.player_count,
	move_robber: cs.tile_count,
	rob_player: cs.player_count,
	discard: cs.resource_type_count,
	place_city: cs.vertex_count,
	place_road: cs.edge_count,
	place_settlement: cs.vertex_count,
	buy_development_card: 1,
	play_knight: 1,
	play_monopoly: cs.resource_type_count,
	play_road_building: 1,
	play_year_of_plenty: cs.resource_type_count * cs.resource_type_count,
	no_action: 1
}
