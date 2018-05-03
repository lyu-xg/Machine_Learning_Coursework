# coding: utf-8
from numpy import *
import sys
import matplotlib.pyplot as pl

#### Use these for 5x5 data
# alphas = (0.0, 0.01, 0.05, 0.1, 0.5, 1)
# epsilons = (0.0, 0.1, 0.25, 0.5, 1)
# lambdas = (0, 0.25, 0.5, 0.75, 1)

alphas = (0.01, 0.05, 0.1, 0.5)
epsilons = (0.1, 0.25, 0.5)
lambdas = (0.25, 0.5, 0.75)

D = {}
for alpha in alphas:
    for epsilon in epsilons:
        data = load('10x10results/alpha{}_epsilon{}.npy'.format(alpha,epsilon))
        D[(alpha, epsilon)] = (data[:,0,:], data[:,1,:])

def fix_epsilon(ep):
    fig = pl.figure(figsize=(12, 9), dpi=100)

    pl.subplot(211)
    for alpha in alphas:
        policy_step, _ = D[(alpha, ep)]
        pl.plot(range(policy_step.shape[1]), mean(policy_step, axis=0))
    pl.legend(['α='+str(a) for a in alphas])
    pl.ylabel('avg steps needed')
    pl.xlabel('episodes')


    pl.subplot(212)
    for alpha in alphas:
        _, q_start = D[(alpha, ep)]
        pl.plot(range(q_start.shape[1]), mean(q_start, axis=0))
    # pl.legend(['α='+str(a) for a in alphas])
    pl.legend(['α='+str(a) for a in alphas])
    pl.ylabel('start state Q value')
    pl.xlabel('episodes')
    pl.show()
    pl.close()


def fix_alpha(a):
    fig = pl.figure(figsize=(12, 9), dpi=100)

    pl.subplot(211)
    for ep in epsilons:
        policy_step, _ = D[(a, ep)]
        pl.plot(range(policy_step.shape[1]), mean(policy_step, axis=0))
    pl.legend(['ε='+str(e) for e in epsilons])
    pl.ylabel('avg steps needed')
    pl.xlabel('episodes')


    pl.subplot(212)
    for ep in epsilons:
        _, q_start = D[(a, ep)]
        pl.plot(range(q_start.shape[1]), mean(q_start, axis=0))
    pl.legend(['ε='+str(e) for e in epsilons])
    pl.ylabel('start state Q value')
    pl.xlabel('episodes')
    pl.show()
    pl.close()

D1 = {}
for alpha in alphas:
    for epsilon in epsilons:
        for l in lambdas:
            data = load('10x10results/alpha{}_epsilon{}_lmbda{}.npy'.format(alpha,epsilon, l))
            D1[(alpha, epsilon, l)] = (data[:,0,:], data[:,1,:])


def fix_alpha_and_ep(a, ep):
    fig = pl.figure(figsize=(12, 9), dpi=100)

    pl.subplot(211)
    for l in lambdas:
        policy_step, _ = D1[(a, ep, l)]
        pl.plot(range(policy_step.shape[1]), mean(policy_step, axis=0))
    pl.legend(['λ='+str(l) for l in lambdas])
    pl.ylabel('avg steps needed')
    pl.xlabel('episodes')


    pl.subplot(212)
    for l in lambdas:
        _, q_start = D1[(a, ep, l)]
        pl.plot(range(q_start.shape[1]), mean(q_start, axis=0))
        print(repr(mean(q_start, axis=0)))
    pl.legend(['λ='+str(l) for l in lambdas])
    pl.ylabel('start state Q value')
    pl.xlabel('episodes')
    pl.show()
    pl.close()


if __name__ == "__main__":
    # fix_epsilon(0.25) # fixing epsilon at 0.25, showing effects of alpha
    # fix_alpha(0.1) # fixing alpha at 0.1, showing effects of epsilon
    fix_alpha_and_ep(0.1, 0.1)

