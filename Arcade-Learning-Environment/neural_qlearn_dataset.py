from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
import numpy as np
import theano
from theano.compat.python2x import OrderedDict
import theano.tensor as T
import copy
from itertools import izip

from random import Random


class NeuralQLearnDataset:
    randGenerator=Random()


    def __init__(self, cnn, history_size = 100000, gamma = .9, alpha = .1, \
                    learning_rate = .5, batch_size = 32):
        """
        Sets up variables for later use.
        
        """
        self.batch_size = batch_size
        self.learning_rate = .5
        self.history_size = history_size
        self.gamma = gamma
        self.alpha = alpha
        self.data = []
        self.actions = []
        self.rewards = []
        self.cnn = cnn
        
        self.counter = 0

    def add(self, data, action, reward):
        """
        Used for adding data to the dataset.

        """
        self.data.append(data.astype('uint8'))
        self.actions.append(action)
        self.rewards.append(reward)

        if len(self.data) > self.history_size:
            self.data.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)

    def has_targets(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_random_dataset(self):
        """
        Creates 80x105x4 tensors, and returns them with targets in a ddm for training. 
        The four channels contain subsequent states. Performs QLearning update step for 
        the target, using the most recent state's action and reward values.

        Arguments:
            None

        Returns:
            DenseDesignMatrix containing a single set of data and target for training.
            Returns None if there is not enough data in the set for a batch.
        """
        
        formatted_data = []
        states = []
        next_states = []
        actions = []
        rewards = []
        states_nparray = []
        
        
        self.counter = 0
        
        #create training batch
        for i in range(self.batch_size):
            #select a random point in history
            data_num = self.randGenerator.randint(3, len(self.data) - 2)
            
            #combine four states for training
            state = np.append(self.data[data_num], 
                            self.data[data_num - 1].astype('float32'), axis=0)
            state = np.append(state, self.data[data_num - 2].astype('float32'), \
                            axis=0)
            state = np.append(state, self.data[data_num - 3].astype('float32'),  \
                            axis=0)
            
            #combine the next four states for Q(s, a)'
            next_state = np.append(self.data[data_num + 1].astype('float32'), \
                            self.data[data_num].astype('float32'), axis=0)
            next_state = np.append(next_state, \
                            self.data[data_num - 1].astype('float32'), axis=0)
            next_state = np.append(next_state, \
                            self.data[data_num - 2].astype('float32'), axis=0)
            
            #put values into lists
            states.append(state)
            next_states.append(next_state)
            actions.append(self.actions[data_num])
            rewards.append(self.rewards[data_num])
   
        #create nparrays
        states_nparray = np.empty((self.batch_size, len(states[0])))
        next_states_nparray = np.empty((self.batch_size, len(next_states[0])))
        for i in range (self.batch_size):
            states_nparray[i] = states[i]
            next_states_nparray[i] = next_states[i]
        
        states = states_nparray.astype('float32')
        next_states = next_states_nparray.astype('float32')
        
        #normalize values
        states /= 256.0
        next_states /= 256.0
        
        #create theano tensors
        states = theano.shared(states, name='input')
        states = T.reshape(states, (self.batch_size, 80, 105, 4))
        states = T.cast(states, dtype='floatX')
            
        next_states = theano.shared(next_states, name='input_max')
        next_states = T.reshape(next_states, (self.batch_size, 80, 105, 4))
        next_states = T.cast(next_states, dtype='floatX')
        
        #get output predictions from nn using the states batch
        q_sa_list = self.cnn.fprop(states).eval()
        
        y_hat = copy.deepcopy(q_sa_list)
        
        #get max of Q(s, a)' for next state batch
        q_sa_prime_list = self.cnn.fprop(next_states).eval()
        
        for i in range(self.batch_size):
            #perform qlearning update to get target value Q(s, a)
            next_state_max = max(q_sa_prime_list[i])
            q_sa_list[i][actions[i]] = q_sa_list[i][actions[i]] + (self.alpha * \
                            (rewards[i] + (self.gamma * next_state_max) - \
                            q_sa_list[i][actions[i]]))
            
        #perform SGD update
        cost = self.cnn.cost(q_sa_list, y_hat)
        params = list(self.cnn.get_params())
        grads = T.grad(cost, params, disconnected_inputs='ignore')
        
        gradients = OrderedDict(izip(params, grads))
        
        update = OrderedDict()

        #perform update
        update.update = (params, [param - self.learning_rate * gradients[param] \
                        for param in params])
        train = theano.function([], updates=update, on_unused_input='ignore',)
        
        for param in params:
            train()
        
    def get_cur_state(self):
        #combine four states
        data = np.append(self.data[len(self.data) - 1], self.data[len(self.data) - 2], axis=0)
        data = np.append(data, self.data[len(self.data) - 3], axis=0)
        data = np.append(data, self.data[len(self.data) - 4], axis=0)
        
        #divide by 256 to make values < 1 for neural net
        data /= 256.0
        
        #create theano variables
        data = theano.shared(data, name='input')
        data = T.reshape(data, (1, 80, 105, 4))
        data = T.cast(data, dtype='floatX')
        return data
        
    def reset_data(self):
        self.data = []
        self.actions = []
        self.rewards = []
        
