from actions.confirm_player_trade.handle import handle_confirm_player_trade
# from actions.confirm_player_trade.mask import mask_confirm_player_trade
from pytan_fast import settings

confirm_player_trade_prefix = "CONFIRM_PLAYER_TRADE"


def get_confirm_player_trade_term(trade_partner_index):
	return confirm_player_trade_prefix,trade_partner_index


def get_confirm_player_trade_mapping():
	confirm_player_trade_mapping = {}
	for trade_partner_index in settings.player_list:
		term = get_confirm_player_trade_term(trade_partner_index)
		callback = (handle_confirm_player_trade, trade_partner_index)
		mask = None #mask_confirm_player_trade(trade_partner_index)
		confirm_player_trade_mapping[term] = {"callback": callback, "mask": mask}
	return confirm_player_trade_mapping
