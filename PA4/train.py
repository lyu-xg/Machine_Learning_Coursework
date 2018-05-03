# -*- coding: utf-8 -*-
from numpy import *
from gridworld import GridworldEnv
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-alg', dest='alg', default='q', type=str)           # q for q-learn, s for sarsa(λ)
parser.add_argument('-size', dest='size', default=5, type=int)           # size of grid world
parser.add_argument('-gamma', dest='gamma', default=0.99, type=float)    # the discount factor
parser.add_argument('-exps', dest='exps', default=500, type=int)         # amount of experiences to run
parser.add_argument('-eps', dest='eps', default=500, type=int)           # amount of episodes per experiment
parser.add_argument('-epsilon', dest='epsilon', default=0.1, type=float) # the epsilon in ε-greedy policy execution
parser.add_argument('-alpha', dest='alpha', default=0.1, type=float)     # the learning rate
parser.add_argument('-lambda', dest='lmbda', default=0, type=float)     # the parameter for Sarsa(λ)
parser.add_argument('-name', dest='name', default='res', type=str)       # the filename to store the graph
args = parser.parse_args()

env = GridworldEnv(shape=[args.size, args.size])

nS = args.size ** 2
nA = 4
initlial_s = nS - args.size 


def optimal_step(Q):
	total_step = 0
	s = initlial_s # initlial state
	terminal = False
	while not terminal:
		# using learned policy
		s, r, terminal = env.move(s, env.get_action(argmax(Q[s])))[0]
		total_step += 1
	return total_step


def run_q_experiment(exp):
	print('Thread {}: running experiment {}'.format(exp, exp))
	Q = zeros((nS, nA))
	policy_step = zeros(args.eps)
	q_start = zeros(args.eps)
	for ep in range(args.eps):
		step = 0		
		s = initlial_s # initlial state is the lower left corner
		while 1:
			# perform action and get new state and reward
			a = argmax(Q[s]) if random.rand() > args.epsilon else random.randint(nA)

			s1, r, termial = env.move(s, env.get_action(a))[0]
			# update the Q value accordingly
			Q[s,a] = (1 - args.alpha) * Q[s,a] + args.alpha * (r + args.gamma * max(Q[s1]))

			s = s1
			step += 1
			if termial or step > nS*1000: break
		policy_step[ep] = optimal_step(Q)
		q_start[ep] = max(Q[initlial_s, :])
	return (policy_step, q_start)

def run_s_experiment(exp):
	print('Thread {}: running experiment {}'.format(exp, exp))
	Q = zeros((nS, nA))
	policy_step = zeros(args.eps)
	q_start = zeros(args.eps)
	for ep in range(args.eps):
		s = initlial_s
		a = 0
		E = zeros((nS, nA)) # eligibility trace table
		terminal = False
		while not terminal:
			s1, r, terminal = env.move(s, env.get_action(a))[0]
			a1 = argmax(Q[s1]) if random.rand() > args.epsilon else random.randint(nA)
			backup_delta = r + args.gamma * Q[s1, a1] - Q[s, a]
			E[s, a] = E[s, a] + 1 # increase eligibility for state we are in
			Q += args.alpha * backup_delta * E # update Q proportional to E
			E = args.gamma * args.lmbda * E # decrease all eligibility at each time step
			s = s1
			a = a1
		policy_step[ep] = optimal_step(Q)
		q_start[ep] = max(Q[initlial_s, :])
	return (policy_step, q_start)


import multiprocessing

pool = multiprocessing.Pool(multiprocessing.cpu_count())
# Every experiment can be ran in parallel because they use different Q table
algo_table = {'q': run_q_experiment, 's': run_s_experiment}
data = pool.map(algo_table[args.alg], range(args.exps))
# save the results of the experiment
save(args.name+'.npy', array(data))
pool.close()

