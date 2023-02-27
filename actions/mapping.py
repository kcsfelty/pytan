from actions.accept_player_trade.mapping import get_accept_player_trade_mapping
from actions.bank_trade.mapping import get_bank_trade_mapping
from actions.buy_development_card.mapping import get_buy_development_card_mapping
from actions.cancel_player_trade.mapping import get_cancel_player_trade_mapping
from actions.confirm_player_trade.mapping import get_confirm_player_trade_mapping
from actions.decline_player_trade.mapping import get_decline_player_trade_mapping
from actions.discard.mapping import get_discard_mapping
from actions.end_turn.mapping import get_end_turn_mapping
from actions.general_port_trade.mapping import get_general_port_trade_mapping
from actions.move_robber.mapping import get_move_robber_mapping
from actions.no_action.mapping import get_no_action_mapping
from actions.offer_player_trade.mapping import get_offer_player_trade_mapping
from actions.place_city.mapping import get_place_city_mapping
from actions.place_road.mapping import get_place_road_mapping
from actions.place_settlement.mapping import get_place_settlement_mapping
from actions.play_knight.mapping import get_play_knight_mapping
from actions.play_monopoly.mapping import get_play_monopoly_mapping
from actions.play_road_building.mapping import get_play_road_building_mapping
from actions.play_year_of_plenty.mapping import get_play_year_of_plenty_mapping
from actions.resource_port_trade.mapping import get_resource_port_trade_mapping
from actions.rob_player.mapping import get_rob_player_mapping
from actions.roll_dice.mapping import get_roll_dice_mapping
from pytan_fast import settings


def get_action_mapping(self):
	action_mapping = {}

	action_mapping.update(get_end_turn_mapping())
	action_mapping.update(get_roll_dice_mapping())

	action_mapping.update(get_bank_trade_mapping(settings.resource_list, settings.bank_trade_ratio))
	action_mapping.update(get_general_port_trade_mapping(settings.resource_list, settings.general_port_trade_ratio))
	action_mapping.update(get_resource_port_trade_mapping(settings.resource_list, settings.resource_port_trade_ratio))

	action_mapping.update(get_offer_player_trade_mapping(settings.resource_list, settings.maximum_player_trade_ratio))
	action_mapping.update(get_accept_player_trade_mapping())
	action_mapping.update(get_decline_player_trade_mapping())
	action_mapping.update(get_cancel_player_trade_mapping())
	action_mapping.update(get_confirm_player_trade_mapping())

	action_mapping.update(get_move_robber_mapping(self.board.tiles))
	action_mapping.update(get_rob_player_mapping())
	action_mapping.update(get_discard_mapping(settings.resource_list))

	action_mapping.update(get_place_city_mapping(self.board.vertices))
	action_mapping.update(get_place_road_mapping(self.board.edges))
	action_mapping.update(get_place_settlement_mapping(self.board.vertices))

	action_mapping.update(get_buy_development_card_mapping())
	action_mapping.update(get_play_knight_mapping())
	action_mapping.update(get_play_monopoly_mapping(settings.resource_list))
	action_mapping.update(get_play_road_building_mapping())
	action_mapping.update(get_play_year_of_plenty_mapping())

	action_mapping.update(get_no_action_mapping())

	return action_mapping


