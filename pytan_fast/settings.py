import numpy as np

from geometry.get_edge_list import get_edge_list
from geometry.get_tile_list import get_tile_list
from geometry.get_vertex_list import get_vertex_list


resource_type_count_tile = 6
development_card_type_count = 5
roll_number_count = 11
port_type_count = 6
tile_list = get_tile_list()
tile_count = len(tile_list)
vertex_count = len(get_vertex_list(tile_list))
edge_count = len(get_edge_list(tile_list))

player_count = 3
player_list = [x for x in range(player_count)]

# Game rules
allow_bank_trading = True
allow_player_trading = True
allow_resource_port_trading = True
allow_general_port_trading = True
settlements_break_road_distance = True
allow_development_card_before_roll = True
only_win_on_turn = True
robber_must_move = True

# Build phase
build_phase_settlement_count = 2
build_phase_free_road = True

# Robber / Robbing
robber_steals_card_quantity = 1
robber_activation_roll = 5
friendly_robber_threshold = 2
rob_player_above_card_count = 7
robber_discard_ratio = 2
robber_discard_round_down = True

# Resources
resource_type_count = 5
resource_list = [x for x in range(resource_type_count)]
resource_card_count_per_type = [19, 19, 19, 19, 19]
settlement_distribute_resource_count = 1
city_distribute_resource_count = 2

# Development Cards
development_card_count_per_type = [14, 2, 2, 2, 5]
max_development_cards_played_per_turn = 1
development_type_count = 5
development_card_list = [x for x in range(development_type_count)]
road_building_road_count = 2
knight_index = 0
monopoly_index = 1
year_of_plenty_index = 2
road_building_index = 3
victory_point_card_index = 4

# Pieces
max_settlement_count = 5
max_city_count = 4
max_road_count = 15
min_settlement_distance = 2

# Board
board_height = 5
board_width = 5
tile_resource_count = [4, 3, 4, 4, 3, 1]
tile_resource_list = [0, 1, 2, 3, 4, 5]
tile_roll_number_type_count = 11
tile_roll_number_count_per_type = [1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 0]
roll_list = [x for x in range(11)]
desert_index = 5

# Costs
settlement_cost = np.array([-1, -1, -1, -1, 0])
city_cost = np.array([0, 0, 0, -2, -3])
road_cost = np.array([-1, -1, 0, 0, 0])
development_card_cost = np.array([0, 0, -1, -1, -1])
play_knight_cost = [1, 0, 0, 0, 0]
play_monopoly_cost = [0, 1, 0, 0, 0]
play_year_of_plenty_cost = [0, 0, 1, 0, 0]
play_road_building_cost = [0, 0, 0, 1, 0]

# Trading
bank_trade_ratio = 4
general_port_trade_ratio = 3
resource_port_trade_ratio = 2
maximum_player_trade_ratio = 2
player_trade_per_turn_limit = 1

# Ports
port_vertex_groups = [
	[(0, 5), (0, 6)],
	[(0, 8), (0, 9)],
	[(1, 10), (1, 11)],
	[(2, 11), (3, 10)],
	[(4, 7), (4, 8)],
	[(5, 3), (5, 4)],
	[(5, 0), (5, 1)],
	[(3, 1), (4, 0)],
	[(1, 3), (2, 2)],
]
port_list = [0, 1, 2, 3, 4, 5]
port_type_count_per_type = [1, 1, 1, 1, 1, 4]
general_port_index = 5

# Victory Points
victory_points_to_win = 10
settlement_victory_points = 1
city_victory_points = 2
victory_point_card_victory_points = 1
longest_road_victory_points = 2
min_longest_road = 5
largest_army_victory_points = 2
min_largest_army = 3
