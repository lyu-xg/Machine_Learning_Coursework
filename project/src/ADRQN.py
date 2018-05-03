import numpy as np
import random
import keras
from keras.models import load_model, Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers import merge, Input
from keras import backend as K
from replay_buffer import ReplayBuffer

# List of hyper-parameters and constants
DISCOUNT = 0.99
NUM_ACTIONS = 6
# Number of frames to throw into network
# NUM_FRAMES = 3

class Qnetwork():
    def __init__(self,h_size,rnn_cell,myScope):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        self.conv1 = slim.convolution2d( \
            inputs=self.imageIn,num_outputs=32,\
            kernel_size=[8,8],stride=[4,4],padding='VALID', \
            biases_initializer=None,scope=myScope+'_conv1')
        self.conv2 = slim.convolution2d( \
            inputs=self.conv1,num_outputs=64,\
            kernel_size=[4,4],stride=[2,2],padding='VALID', \
            biases_initializer=None,scope=myScope+'_conv2')
        self.conv3 = slim.convolution2d( \
            inputs=self.conv2,num_outputs=64,\
            kernel_size=[3,3],stride=[1,1],padding='VALID', \
            biases_initializer=None,scope=myScope+'_conv3')
        self.conv4 = slim.convolution2d( \
            inputs=self.conv3,num_outputs=h_size,\
            kernel_size=[7,7],stride=[1,1],padding='VALID', \
            biases_initializer=None,scope=myScope+'_conv4')
        
        self.trainLength = tf.placeholder(dtype=tf.int32)
        #We take the output from the final convolutional layer and send it to a recurrent layer.
        #The input must be reshaped into [batch x trace x units] for rnn processing, 
        #and then returned to [batch x units] when sent through the upper levles.
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        self.convFlat = tf.reshape(slim.flatten(self.conv4),[self.batch_size,self.trainLength,h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
                inputs=self.convFlat,cell=rnn_cell,dtype=tf.float32,initial_state=self.state_in,scope=myScope+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        #The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA,self.streamV = tf.split(self.rnn,2,1)
        self.AW = tf.Variable(tf.random_normal([h_size//2,4]))
        self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        self.salience = tf.gradients(self.Advantage,self.imageIn)
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,4,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        
        #In order to only propogate accurate gradients through the network, we will mask the first
        #half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
        self.maskB = tf.ones([self.batch_size,self.trainLength//2])
        self.mask = tf.concat([self.maskA,self.maskB],1)
        self.mask = tf.reshape(self.mask,[-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class RecurQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
        cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
        self.mainQN = Qnetwork(h_size,cell,'main')
        self.targetQN = Qnetwork(h_size,cellT,'target')


    def predict_movement(self, s, state, epsilon):
        a = 0
        if np.random.rand(1) < epsilon:
            state = sess.run(self.mainQN.rnn_state,\
                feed_dict={self.mainQN.scalarInput:[s],self.mainQN.trainLength:1,self.mainQN.state_in:state,self.mainQN.batch_size:1})
            a = np.random.randint(0,NUM_ACTIONS)
        else:
            a, state = sess.run([self.mainQN.predict,self.mainQN.rnn_state],\
                feed_dict={self.mainQN.scalarInput:[s],self.mainQN.trainLength:1,self.mainQN.state_in:state,self.mainQN.batch_size:1})
            a = a[0]
        return a, state

    def train(self, train_batch):
        """Trains network to fit given parameters"""
        state_train = (np.zeros([batch_size,h_size]),np.zeros([batch_size,h_size]))
        Q1 = sess.run(self.mainQN.predict,feed_dict={\
            self.mainQN.scalarInput:np.vstack(train_batch[:,3]),\
            self.mainQN.trainLength:trace_length,self.mainQN.state_in:state_train,self.mainQN.batch_size:batch_size})
        Q2 = sess.run(self.targetQN.Qout,feed_dict={\
            self.targetQN.scalarInput:np.vstack(train_batch[:,3]),\
            self.targetQN.trainLength:trace_length,self.targetQN.state_in:state_train,self.targetQN.batch_size:batch_size})
        end_multiplier = -(train_batch[:,4] - 1)
        doubleQ = Q2[range(batch_size*trace_length),Q1]
        targetQ = train_batch[:,2] + (y*doubleQ * end_multiplier)
        #Update the network with our target values.
        sess.run(self.mainQN.updateModel, \
            feed_dict={self.mainQN.scalarInput:np.vstack(train_batch[:,0]),self.mainQN.targetQ:targetQ,\
            self.mainQN.actions:train_batch[:,1],self.mainQN.trainLength:trace_length,\
            self.mainQN.state_in:state_train,self.mainQN.batch_size:batch_size})

        # Print the loss every 10 iterations.
        # if observation_num % 10 == 0:
        #     print("We had a loss equal to ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Weights saved.")

    def load_network(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print("Weights loaded.")

    def target_train(self):
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)
