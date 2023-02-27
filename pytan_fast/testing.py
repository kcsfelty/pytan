import numpy as np

def set_intersection(listA, listB):
	return [a for a in listA if a in listB]


	
	










observation_length = sum([observation_lengths[key] for key in observation_lengths])
observation = np.zeros((player_count, observation_length))
observation_indices = {}

observation_slices = {}
current_index = 0
for term in observation_lengths:
	next_index = current_index + observation_lengths[term]
	observation_slices[term] = observation[:, current_index:next_index].view()
	current_index = next_index

# Reshape anything which is one-hot encoded
observation_slices[tile_resource].shape = (player_count, tile_count, resource_types)
observation_slices[tile_roll_number].shape = (player_count, tile_count, roll_numbers)
observation_slices[vertex_owner].shape = (player_count, vertex_count, player_count)
observation_slices[edge_owner].shape = (player_count, edge_count, player_count)

# Construct additional helper matrices
bank_harvest = np.zeros((roll_numbers, resource_types))
players_harvest = np.zeros((roll_numbers, resource_types, player_count))
vertex_harvest = np.zeros((vertex_count, roll_numbers, resource_types))


board = Geometry(observation_slices)

#print(observation.T)
#print(board.tiles[0].has_robber)
#board.tiles[0].has_robber[:] = 1
#print(observation.T)
#print(board.tiles[0].has_robber)
#print(board.tiles[0].resource)

#print(vertex_harvest[0][6][:])
vertex_harvest[0][6][:] = [1, 0, 0, 0, 0]
vertex_harvest[0][4][:] = [0, 0, 0, 2, 0]
#print(vertex_harvest[0][6][:])
#print(vertex_harvest[0])

roll_matrix = np.identity(roll_numbers)
roll_results = np.dot(roll_matrix, vertex_harvest)

print(roll_results[0])





















