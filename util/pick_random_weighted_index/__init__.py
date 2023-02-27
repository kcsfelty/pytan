import random


def pick_random_weighted_index(deck):
	indices = []
	for i in range(len(deck)):
		for j in range(deck[i]):
			indices.append(i)
	if sum(indices) == 0:
		return False
	random.shuffle(indices)
	index = random.randint(0, len(indices) - 1)
	return int(indices[index])
