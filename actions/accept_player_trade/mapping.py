from actions.accept_player_trade.handle import handle_accept_player_trade
# from actions.accept_player_trade.mask import mask_accept_player_trade

accept_player_trade = "ACCEPT_PLAYER_TRADE"


def get_accept_player_trade_mapping():
	return {(accept_player_trade, None): {"callback": (handle_accept_player_trade, None),}}# "mask": mask_accept_player_trade()}}
