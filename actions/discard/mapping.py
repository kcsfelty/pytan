from actions.discard.handle import handle_discard
# from actions.discard.mask import mask_discard

discard_prefix = "DISCARD"


def get_discard_term(trade):
	return discard_prefix, str(trade)


def get_discard_mapping(resource_list):
	discard_mapping = {}
	for resource_index in resource_list:
		trade = [0 for _ in resource_list]
		trade[resource_index] = -1
		term = get_discard_term(trade)
		callback = (handle_discard, trade)
		mask = None #mask_discard(trade)
		discard_mapping[term] = {"callback": callback, "mask": mask}
	return discard_mapping
