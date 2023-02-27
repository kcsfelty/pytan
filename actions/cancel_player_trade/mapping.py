from actions.cancel_player_trade.handle import handle_cancel_player_trade
# from actions.cancel_player_trade.mask import mask_cancel_player_trade

cancel_player_trade = "CANCEL_PLAYER_TRADE"


def get_cancel_player_trade_mapping():
	return {(cancel_player_trade, None): {"callback": (handle_cancel_player_trade, None),}} #"mask": mask_cancel_player_trade()}}
