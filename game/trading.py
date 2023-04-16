import numpy as np

from util.get_heterogeneous_trade_ratio_list import get_heterogeneous_trade_ratio_list
from util.get_homogeneous_trade_ratio_list import get_homogeneous_trade_ratio_list
import reference.settings as gs


def get_trades():
	bank_trades_base = get_homogeneous_trade_ratio_list(gs.resource_list, gs.bank_trade_ratio)
	general_port_trades_base = get_homogeneous_trade_ratio_list(gs.resource_list, gs.general_port_trade_ratio)
	resource_port_trades_base = get_homogeneous_trade_ratio_list(gs.resource_list, gs.resource_port_trade_ratio)
	player_trades_base = get_heterogeneous_trade_ratio_list(gs.resource_list, gs.maximum_player_trade_ratio)
	year_of_plenty_trades_base = []
	discard_trades_base = []
	for x in gs.resource_list:
		trade = np.zeros(gs.resource_type_count, dtype=np.int8)
		trade[x] = -1
		discard_trades_base.append(trade)
		for y in gs.resource_list:
			trade = np.zeros(gs.resource_type_count, dtype=np.int8)
			trade[x] += 1
			trade[y] += 1
			year_of_plenty_trades_base.append(trade)
	return bank_trades_base, general_port_trades_base, resource_port_trades_base, player_trades_base, year_of_plenty_trades_base, discard_trades_base


def get_trade_lookup(trade_list, total_cards=5):
	all_hands = np.mgrid[0:total_cards, 0:total_cards, 0:total_cards, 0:total_cards, 0:total_cards].T
	all_trades = np.expand_dims(trade_list, axis=(1, 2, 3, 4, 5))
	return np.all(all_hands + all_trades >= 0, axis=-1).T


bank_trades, general_port_trades, resource_port_trades, player_trades, year_of_plenty_trades, discard_trades = get_trades()
bank_trades = np.array(bank_trades)
general_port_trades = np.array(general_port_trades)
resource_port_trades = np.array(resource_port_trades)
player_trades = np.array(player_trades)
year_of_plenty_trades = np.array(year_of_plenty_trades)
bank_trades_lookup_bank = get_trade_lookup(bank_trades * -1)
general_port_trades_lookup_bank = get_trade_lookup(general_port_trades * -1)
resource_port_trades_lookup_bank = get_trade_lookup(resource_port_trades * -1)
bank_trades_lookup = get_trade_lookup(bank_trades)
general_port_trades_lookup = get_trade_lookup(general_port_trades)
resource_port_trades_lookup = get_trade_lookup(resource_port_trades)
player_trades_lookup = get_trade_lookup(player_trades)
discard_trades_lookup = get_trade_lookup(discard_trades)
