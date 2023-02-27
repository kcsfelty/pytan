def list_hash(input_list):
	lookup = {}
	for i in range(len(input_list)):
		lookup[input_list[i]] = i
	return lookup
