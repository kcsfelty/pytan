from reference.game_state import game_state_degrees
from reference.private_state import private_state_degrees
from reference.public_state import public_state_degrees
import reference.definitions as df
import numpy as np


class ObservationMetrics:
	def __init__(self):
		self.game_terms = []
		for term in game_state_degrees:
			if game_state_degrees[term] > 1:
				self.game_terms.extend([term + "_" + str(i) for i in range(game_state_degrees[term])])
			else:
				self.game_terms.append(term)
		self.private_terms = []
		for term in private_state_degrees:
			if private_state_degrees[term] > 1:
				self.private_terms.extend([term + "_" + str(i) for i in range(private_state_degrees[term])])
			else:
				self.private_terms.append(term)
		self.public_terms = []
		for term in public_state_degrees:
			if public_state_degrees[term] > 1:
				self.public_terms.extend([term + "_" + str(i) for i in range(public_state_degrees[term])])
			else:
				self.public_terms.append(term)

		self.game_state_len = sum([game_state_degrees[term] for term in game_state_degrees])
		self.private_state_len = sum([private_state_degrees[term] for term in private_state_degrees])
		self.public_state_len = sum([public_state_degrees[term] for term in public_state_degrees])

		# Game metrics
		self.turn_count_index = self.game_terms.index(df.turn_number)

		# Player metrics
		self.victory_points_index = self.public_terms.index(df.victory_points)
		self.settlement_count_index = self.public_terms.index(df.settlement_count)
		self.city_count_index = self.public_terms.index(df.city_count)
		self.road_count_index = self.public_terms.index(df.road_count)
		self.longest_road_index = self.public_terms.index(df.longest_road)
		self.owns_longest_road_index = self.public_terms.index(df.owns_longest_road)
		self.owns_largest_army_index = self.public_terms.index(df.owns_largest_army)
		self.knight_cards_played_index = self.public_terms.index(df.development_cards_played + "_0")
		self.monopoly_cards_played_index = self.public_terms.index(df.development_cards_played + "_1")
		self.year_of_plenty_cards_played_index = self.public_terms.index(df.development_cards_played + "_2")
		self.road_building_cards_played_index = self.public_terms.index(df.development_cards_played + "_3")
		self.victory_point_cards_played_index = self.public_terms.index(df.development_cards_played + "_4")

	def summarize(self, observation):
		game_state = observation[:, :self.game_state_len]
		public_state = observation[:, self.game_state_len + self.private_state_len:]
		public_state = np.reshape(public_state, (-1, self.public_state_len))

		game_metrics = {
			df.turn_number: game_state[:, self.turn_count_index]}

		player_metrics = {
			df.settlement_count: public_state[:, self.settlement_count_index],
			df.city_count: public_state[:, self.city_count_index],
			df.road_count: public_state[:, self.road_count_index],
			df.longest_road: public_state[:, self.longest_road_index],
			df.owns_longest_road: public_state[:, self.owns_longest_road_index],
			df.owns_largest_army: public_state[:, self.owns_largest_army_index],
			"knight_cards_played": public_state[:, self.knight_cards_played_index],
			"monopoly_cards_played": public_state[:, self.monopoly_cards_played_index],
			"year_of_plenty_cards_played": public_state[:, self.year_of_plenty_cards_played_index],
			"road_building_cards_played": public_state[:, self.road_building_cards_played_index],
			"victory_point_cards_played": public_state[:, self.victory_point_cards_played_index]}

		return game_metrics, player_metrics
