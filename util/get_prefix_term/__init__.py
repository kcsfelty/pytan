def get_prefix_term(prefix, location_list, delimiter="_"):
	return ["{}{}{}".format(prefix, delimiter, location) for location in location_list]
