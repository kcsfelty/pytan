from actions.bank_trade.handle import handle_bank_trade
# from actions.bank_trade.mask import mask_bank_trade
from util.get_homogeneous_trade_ratio_list import get_homogeneous_trade_ratio_list

bank_trade_prefix = "BANK_TRADE"


def get_bank_trade_term(trade):
	return bank_trade_prefix, str(trade)


def get_bank_trade_mapping(resource_list, bank_trade_ratio):
	trades = get_homogeneous_trade_ratio_list(resource_list, bank_trade_ratio)
	bank_trade_mapping = {}
	for trade in trades:
		term = get_bank_trade_term(trade)
		callback = (handle_bank_trade, trade)
		mask = None#mask_bank_trade(trade)
		bank_trade_mapping[term] = {"callback": callback, "mask": mask}
	return bank_trade_mapping
