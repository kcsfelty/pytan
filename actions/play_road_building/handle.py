from pytan_fast import settings


def handle_play_road_building(_, player_index, self):
	self.play_development_card(player_index, settings.road_building_index)
	self.road_builder_block_state = False
	self.road_builder_count = 0
	self.road_builder_responsible_index = player_index
