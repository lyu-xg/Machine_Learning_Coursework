import argparse
from agent import Agent
import sys

# Hyperparameters
MODEL = 'DDQN' if len(sys.argv) < 2 else sys.argv[1]
STEPS_TODO = 100000 if len(sys.argv) < 3 else int(sys.argv[2])

agent = Agent(MODEL)
agent.load_network('saved/{}.weights'.format(MODEL))
agent.train(STEPS_TODO)

# parser = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

# Parse arguments
# parser.add_argument("-n", "--network", type=str, action='store', help="Please specify the network you wish to use, either DQN or DDQN", required=True)
# parser.add_argument("-m", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)
# parser.add_argument("-l", "--load", type=str, action='store', help="Please specify the file you wish to load weights from(for example saved.h5)", required=False)
# parser.add_argument("-s", "--save", type=str, action='store', help="Specify folder to render simulation of network in", required=False)
# parser.add_argument("-x", "--statistics", action='store_true', help="Specify to calculate statistics of network(such as average score on game)", required=False)
# parser.add_argument("-v", "--view", action='store_true', help="Display the network playing a game of space-invaders. Is overriden by the -s command", required=False)

# args = parser.parse_args()
# print(args)


# agent.load_network('SpaceInvaders-v0_DQN.h5')

# agent.simulate()

# if args.load:
#     agent.load_network(args.load)

# if args.mode == "train":
#     agent.train(STEPS_TODO)

# if args.statistics:
#     stat = agent.calculate_mean()
#     print("Game Statistics")
#     print(stat)

# if args.save:
#     agent.simulate(path=args.save, save=True)
# elif args.view:
#     agent.simulate()

