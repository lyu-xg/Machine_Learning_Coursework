# -*- coding: utf-8 -*-
import os

os.system('mkdir results')

for alpha in (0.0, 0.01, 0.05, 0.1, 0.5, 1):
	for epsilon in (0.0, 0.1, 0.25, 0.5, 1):
		cmd = 'python train.py -alpha {} -epsilon {} -name alpha{}_epsilon{}'.format(
			alpha, epsilon, alpha, epsilon)
		print('running "{}"'.format(cmd))
		os.system(cmd)


for lmbda in (0.1,):# 0.25, 0.75, 1):
	for alpha in (0.0, 0.01, 0.05, 0.1, 0.5, 1):
		for epsilon in (0.0, 0.1, 0.25, 0.5, 1):
			cmd = 'python train.py -alpha {} -epsilon {} -lambda {} -name alpha{}_epsilon{}_lmbda{}'.format(
				alpha, epsilon, lmbda, alpha, epsilon, lmbda)
			print('running "{}"'.format(cmd))
			os.system(cmd)

for alpha in (0.0, 0.01, 0.05, 0.1, 0.5, 1):
	for epsilon in (0.0, 0.1, 0.25, 0.5, 1):
		for lmbda in (0, 0.25, 0.5, 0.75, 1):
			cmd = 'python q_learn.py -alpha {} -epsilon {} -lambda {} -name alpha{}_epsilon{}_lmbda{}'.format(
				alpha, epsilon, lmbda, alpha, epsilon, lmbda)
			print('running "{}"'.format(cmd))
			os.system(cmd)
