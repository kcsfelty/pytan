from actions.play_road_building.handle import handle_play_road_building
# from actions.play_road_building.mask import mask_play_road_building

play_road_building = "PLAY_ROAD_BUILDING"


def get_play_road_building_mapping():
	return {(play_road_building, None): {"callback": (handle_play_road_building, None),}}# "mask": mask_play_road_building()}}
