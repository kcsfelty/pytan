from actions.general_port_trade.handle import handle_general_port_trade
# from actions.general_port_trade.mask import mask_general_port_trade
from util.get_homogeneous_trade_ratio_list import get_homogeneous_trade_ratio_list

general_port_trade_prefix = "GENERAL_PORT_TRADE"


def get_general_port_trade_term(trade):
	return general_port_trade_prefix, str(trade)


def get_general_port_trade_mapping(resource_list, general_port_trade_ratio):
	trades = get_homogeneous_trade_ratio_list(resource_list, general_port_trade_ratio)
	general_port_trade_mapping = {}
	for trade in trades:
		term = get_general_port_trade_term(trade)
		callback = (handle_general_port_trade, trade)
		mask = None# mask_general_port_trade(trade)
		general_port_trade_mapping[term] = {"callback": callback, "mask": mask}
	return general_port_trade_mapping
