import gym
import cv2
import numpy as np
from collections import deque
from DRQN import RecurQ
import time
import random
# List of hyper-parameters and constants
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 32
TOT_FRAME = 1000000
EPSILON_DECAY = 300000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1.0
H_SIZE = 512
TRACE_LENGTH = 8
BATCH_SIZE = 4
GAME = 'SpaceInvaders-v0'

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, episode_buffer):
        while self.buffer_size - self.count < len(episode_buffer):
            out = self.buffer.popleft()
            self.count -= len(out)
        self.buffer.append(episode_buffer)
        self.count += len(episode_buffer)

    def size(self):
        return self.count

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer,batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size * trace_length, 5]) # each transition shape is (5,)

    def clear(self):
        self.buffer.clear()
        self.count = 0


class Agent(object):

    def __init__(self, mode):
        self.mode = mode
        self.env = gym.make(GAME)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.deep_q = RecurQ()

    def load_network(self, path):
        self.deep_q.load_network(path)

    def img_clean(self, img):
        # resize the image to 84x84x3 (also chopping the margins)
        return cv2.resize(img, (84, 90))[1:85, :, :]/255.0


    def train(self, steps_todo):
        step_num = 0
        s = self.img_clean(self.env.reset())
        epsilon = INITIAL_EPSILON

        single_episode_step_num = 0
        single_episode_reward = 0
        single_episode_experience_buffer = []

        hidden_state = (np.zeros([1,H_SIZE]),np.zeros([1,H_SIZE]))

        while step_num < steps_todo:
            if not step_num % 1000:
                print(("Executing loop %d" %step_num))

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            a, hidden_state = self.deep_q.predict_movement(s, hidden_state, epsilon)

            s1, r, done, _ = self.env.step(a)
            s1 = self.img_clean(s1)

            if done:
                print("[{}/{}]Lived with maximum time ".format(step_num,steps_todo), single_episode_step_num)
                print("[{}/{}]Earned a total of reward equal to ".format(step_num,steps_todo), single_episode_reward)
                self.env.reset()
                ep_buf = list(zip(np.array(single_episode_experience_buffer)))
                self.replay_buffer.add(ep_buf)
                single_episode_step_num = 0
                single_episode_reward = 0

            single_episode_experience_buffer.append(np.reshape(np.array([s, a, r, s1, d]),[1,5]))
            single_episode_reward += r

            if self.replay_buffer.size() > MIN_OBSERVATION:
                train_batch = self.replay_buffer.sample(MINIBATCH_SIZE)
                self.deep_q.train(train_batch)
                self.deep_q.target_train()

            # Save the network every 100000 iterations
            if step_num % 10000 == 9999:
                print("Saving Network")
                self.deep_q.save_network("saved/{}.weights".format(self.mode))

            single_episode_step_num += 1
            step_num += 1
            s = s1

    def simulate(self, path = "", save = False):
        """Simulates game"""
        done = False
        tot_award = 0
        if save:
            self.env.monitor.start(path, force=True)
        self.env.reset()
        self.env.render()
        while not done:
            state = self.convert_process_buffer()
            predict_movement = self.deep_q.predict_movement(state, 0)[0]
            self.env.render()
            time.sleep(0.001)
            observation, reward, done, _ = self.env.step(predict_movement)
            tot_award += reward
            self.process_buffer.append(observation)
            self.process_buffer = self.process_buffer[1:]
        if save:
            self.env.monitor.close()

    def calculate_mean(self, num_samples = 100):
        reward_list = []
        print("Printing scores of each trial")
        for i in range(num_samples):
            done = False
            tot_award = 0
            self.env.reset()
            while not done:
                state = self.convert_process_buffer()
                predict_movement = self.deep_q.predict_movement(state, 0.0)[0]
                observation, reward, done, _ = self.env.step(predict_movement)
                tot_award += reward
                self.process_buffer.append(observation)
                self.process_buffer = self.process_buffer[1:]
            print(tot_award)
            reward_list.append(tot_award)
        return np.mean(reward_list), np.std(reward_list)

