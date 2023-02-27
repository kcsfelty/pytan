from actions.resource_port_trade.handle import handle_resource_port_trade
# from actions.resource_port_trade.mask import mask_resource_port_trade
from util.get_homogeneous_trade_ratio_list import get_homogeneous_trade_ratio_list

resource_port_trade_prefix = "RESOURCE_PORT_TRADE"


def get_resource_port_trade_term(trade):
	return resource_port_trade_prefix, str(trade)


def get_resource_port_trade_mapping(resource_list, resource_port_trade_ratio):
	trades = get_homogeneous_trade_ratio_list(resource_list, resource_port_trade_ratio)
	resource_port_trade_mapping = {}
	for trade in trades:
		term = get_resource_port_trade_term(trade)
		callback = (handle_resource_port_trade, trade)
		mask = None#mask_resource_port_trade(trade)
		resource_port_trade_mapping[term] = {"callback": callback, "mask": mask}
	return resource_port_trade_mapping
