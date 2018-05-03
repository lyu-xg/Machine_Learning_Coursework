from agent import Agent
import sys

model, _ = sys.argv[1].split('/')[-1].split('.')

agent = Agent(model)

agent.load_network(sys.argv[1])

agent.simulate()
