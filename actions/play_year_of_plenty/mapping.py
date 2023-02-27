from actions.play_year_of_plenty.handle import handle_play_year_of_plenty
# from actions.play_year_of_plenty.mask import mask_play_year_of_plenty
from pytan_fast import settings

play_year_of_plenty_prefix = "PLAY_YEAR_OF_PLENTY"


def get_play_year_of_plenty_term(trade):
	return play_year_of_plenty_prefix, str(trade)


def get_play_year_of_plenty_mapping():
	trades = []
	for x in settings.resource_list:
		for y in settings.resource_list:
			trade = [0 for _ in settings.resource_list]
			trade[x] += 1
			trade[y] += 1
			trades.append(trade)
	play_year_of_plenty_mapping = {}
	for trade in trades:
		term = get_play_year_of_plenty_term(trade)
		callback = (handle_play_year_of_plenty, trade)
		mask = None #mask_play_year_of_plenty(trade)
		play_year_of_plenty_mapping[term] = {"callback": callback, "mask": mask}
	return play_year_of_plenty_mapping
