import numpy as np

from util.get_heterogeneous_trade_ratio_list import get_heterogeneous_trade_ratio_list
from util.get_homogeneous_trade_ratio_list import get_homogeneous_trade_ratio_list
import pytan_fast.settings as gs


def get_trades():
	bank_trades = get_homogeneous_trade_ratio_list(gs.resource_list, gs.bank_trade_ratio)
	general_port_trades = get_homogeneous_trade_ratio_list(gs.resource_list, gs.general_port_trade_ratio)
	resource_port_trades = get_homogeneous_trade_ratio_list(gs.resource_list, gs.resource_port_trade_ratio)
	player_trades = get_heterogeneous_trade_ratio_list(gs.resource_list, gs.maximum_player_trade_ratio)
	year_of_plenty_trades = []
	discard_trades = []
	for x in gs.resource_list:
		trade = np.zeros(gs.resource_type_count, dtype=np.int8)
		trade[x] = -1
		discard_trades.append(trade)
		for y in gs.resource_list:
			trade = np.zeros(gs.resource_type_count, dtype=np.int8)
			trade[x] += 1
			trade[y] += 1
			year_of_plenty_trades.append(trade)
	return bank_trades, general_port_trades, resource_port_trades, player_trades, year_of_plenty_trades, discard_trades
