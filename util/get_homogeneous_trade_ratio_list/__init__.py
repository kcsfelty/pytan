import numpy as np
import pytan_fast.settings as gs


def get_homogeneous_trade_ratio_list(resource_list, ratio):
	valid_trade_list = []
	for export_resource in resource_list:
		for import_resource in resource_list:
			if export_resource is not import_resource:
				trade = np.zeros(gs.resource_type_count, dtype=np.int8)
				trade[export_resource] = -1 * ratio
				trade[import_resource] = 1
				valid_trade_list.append(trade)
	return valid_trade_list
