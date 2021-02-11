# Testing out knowledge class

from knowledge import *
from agent import Agent

agent = Agent()
knowledge = Knowledge(agent)

print(knowledge.agent.name)
print(knowledge.move())