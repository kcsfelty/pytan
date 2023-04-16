import numpy as np
import reference.settings as gs

def get_heterogeneous_trade_ratio_list(resource_list, maximum_ratio):
	base_trades = []
	for export_resource in resource_list:
		for import_resource in resource_list:
			if export_resource is not import_resource:
				trade = np.zeros(gs.resource_type_count, dtype=np.int8)
				trade[export_resource] = -1
				trade[import_resource] = +1
				base_trades.append(trade)
	compound_trades = []
	for _ in range(maximum_ratio - 1):
		for export_resource in resource_list:
			for trade in base_trades:
				if trade[export_resource] != +1:
					new_trade = [x for x in trade]
					new_trade[export_resource] -= 1
					if new_trade not in compound_trades:
						compound_trades.append(new_trade)
		base_trades.extend(compound_trades)
		compound_trades = []
	return base_trades
