import random


def reverse_histogram(weight_list):
	index_list = []
	for i in range(len(weight_list)):
		for count in range(weight_list[i]):
			index_list.append(i)
	random.shuffle(index_list)
	return index_list
