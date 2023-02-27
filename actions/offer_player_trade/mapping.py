from actions.offer_player_trade.handle import handle_offer_player_trade
# from actions.offer_player_trade.mask import mask_offer_player_trade
from util.get_heterogeneous_trade_ratio_list import get_heterogeneous_trade_ratio_list

offer_player_trade_prefix = "OFFER_PLAYER_TRADE"


def get_offer_player_trade_term(trade):
	return offer_player_trade_prefix, str(trade)


def get_offer_player_trade_mapping(resource_list, maximum_player_trade_ratio):
	trades = get_heterogeneous_trade_ratio_list(resource_list, maximum_player_trade_ratio)
	offer_player_trade_mapping = {}
	for trade in trades:
		term = get_offer_player_trade_term(trade)
		callback = (handle_offer_player_trade, trade)
		mask = None #mask_offer_player_trade(trade)
		offer_player_trade_mapping[term] = {"callback": callback, "mask": mask}
	return offer_player_trade_mapping

