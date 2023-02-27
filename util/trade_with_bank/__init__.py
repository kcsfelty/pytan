from game import game_settings
from observations.game_observation.game_observation_definitions import get_bank_resources_term
from observations.private_observation.private_observation_definitions import get_private_resources_term


def trade_with_bank(observation, lookup, trade):
	game_observation, private_observation, public_observation = observation
	game_lookup, private_lookup, public_lookup = lookup
	for resource in game_settings.resource_list:
		game_observation[game_lookup[get_bank_resources_term(resource)]] -= trade[resource]
		private_observation[private_lookup[get_private_resources_term(resource)]] += trade[resource]
