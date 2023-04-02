import numpy as np

from pytan_fast.get_trades import get_trades


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
