from actions.decline_player_trade.handle import handle_decline_player_trade
# from actions.decline_player_trade.mask import mask_decline_player_trade

decline_player_trade = "DECLINE_PLAYER_TRADE"


def get_decline_player_trade_mapping():
	return {(decline_player_trade, None): {"callback": (handle_decline_player_trade, None),}}# "mask": mask_decline_player_trade()}}
