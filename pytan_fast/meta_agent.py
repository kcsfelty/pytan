from tf_agents.trajectories import TimeStep


class MetaAgent:
	def __init__(self, agent_list, game_count):
		self.agent_list = agent_list
		self.agent_count = len(self.agent_list)
		self.game_count = game_count

	def train(self):
		for agent in self.agent_list:
			agent.train()

	def act(self, time_step_list):
		action_list = []
		time_step_list = [TimeStep(*time_step_list[i:i+4]) for i in range(0, len(time_step_list), 4)]
		for agent, time_step in zip(self.agent_list, time_step_list):
			action_list.append(agent.act(time_step))
		return action_list
