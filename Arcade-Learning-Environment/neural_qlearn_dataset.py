from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
import numpy as np
import theano
from theano.compat.python2x import OrderedDict
from pylearn2.utils import safe_zip
import theano.tensor as T
import copy
from itertools import izip

from random import Random


class NeuralQLearnDataset:
    randGenerator=Random()


    def __init__(self, cnn, batch_size, history_size = 100000, \
                    gamma = .9, alpha = .8, learning_rate = .5):
        """
        Sets up variables for later use.
        
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.history_size = history_size
        self.gamma = gamma
        self.alpha = alpha
        self.data = []
        self.actions = []
        self.rewards = []
        self.cnn = cnn
        self.train_mini_batches = False

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

    def train(self):
        """
        Creates 80x105x4 tensors and trains for one epoch with them. 
        The four channels contain subsequent images. Performs QLearning 
        update step for the target, using the most recent image's action 
        and reward values.
        """
        
        formatted_data = []
        states = []
        next_states = []
        actions = []
        rewards = []
        states_nparray = []
        
        #create training batch
        for i in range(self.batch_size):
            #select a random point in history
            data_num = self.randGenerator.randint(3, len(self.data) - 2)
            
            state = np.empty(33600)
            next_state = np.empty(33600)
            
            #combine four states for training
            state[:8400] = self.data[data_num - 3].astype('float32')
            state[8400:(8400 * 2)] = self.data[ 
                                        data_num - 2].astype('float32')
            state[(8400 * 2):(8400 * 3)] = self.data[ 
                                        data_num - 1].astype('float32')
            state[(8400 * 3):] = self.data[data_num].astype('float32')
            
            #combine the next four states for Q(s, a)'
            state[:8400] = self.data[data_num - 4].astype('float32')
            state[8400:(8400 * 2)] = self.data[data_num - 3].astype('float32')
            state[(8400 * 2):(8400 * 3)] = self.data[ 
                                        data_num - 2].astype('float32')
            state[(8400 * 3):] = self.data[data_num - 1].astype('float32')
            
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
        states = theano.shared(states, name='input', borrow=True)
        states = T.reshape(states, (self.batch_size, 80, 105, 4))
        states = T.cast(states, dtype='floatX')
            
        next_states = theano.shared(next_states, name='input_max', borrow=True)
        next_states = T.reshape(next_states, (self.batch_size, 80, 105, 4))
        next_states = T.cast(next_states, dtype='floatX')
        
        #get output predictions from nn using the states batch
        q_sa_list = self.cnn.fprop(states).eval()
        
        y_hat = copy.deepcopy(q_sa_list)
        
        #get max of Q(s, a)' for next state batch
        q_sa_prime_list = self.cnn.fprop(next_states).eval()
        
        print q_sa_list[0]
        print q_sa_list[1]
        print q_sa_list[2]
        
        for i in range(self.batch_size):
            #perform qlearning update to get target value Q(s, a)
            next_state_max = max(q_sa_prime_list[i])
            y_hat[i][actions[i]] = q_sa_list[i][actions[i]] + (self.alpha * \
                            (rewards[i] + (self.gamma * next_state_max) - \
                            q_sa_list[i][actions[i]]))
            
        #perform SGD update
        num_mini_batches = self.batch_size / 32
        
        if self.train_mini_batches:
            for i in range(num_mini_batches - 1):
                cost = self.cnn.cost(q_sa_list[i * num_mini_batches : i + 1 * \
                                                    num_mini_batches] , \
                                                    y_hat[i * num_mini_batches : \
                                                    i + 1 * num_mini_batches])
                
                params = list(self.cnn.get_params())
                grads = T.grad(cost, params, disconnected_inputs='ignore')
            
                gradients = OrderedDict(izip(params, grads))
            
                update = OrderedDict()

                #perform update
                update.update = (dict(safe_zip(params, [param - self.learning_rate * 
                                            gradients[param] for param in params])))
                train = theano.function([], updates=update, on_unused_input='ignore',)
            
                train()
        else:
            cost = self.cnn.cost(q_sa_list, y_hat)
            
            params = list(self.cnn.get_params())
            grads = T.grad(cost, params, disconnected_inputs='ignore')
            
            #for vals in grads[0].eval():
             #   for val in vals[0][0]:
              #      print val
            
        
            gradients = OrderedDict(izip(params, grads))
        
            update = OrderedDict()

            #perform update
            update.update = (dict(safe_zip(params, [param - self.learning_rate * 
                                        gradients[param] for param in params])))
            train = theano.function([], updates=update, on_unused_input='ignore')
        
            train()
        
        q_sa_list2 = self.cnn.fprop(states).eval()
        
        print "----------------------------"
        
        print q_sa_list2[0]
        print q_sa_list2[1]
        print q_sa_list2[2]
        
        
    def get_cur_state(self):
        #combine last four states
        state = np.empty(33600)
            
        state[:8400] = self.data[len(self.data) - 1].astype('float32')
        state[8400:(8400 * 2)] = self.data[len(self.data) - 2].astype('float32')
        state[(8400 * 2):(8400 * 3)] = self.data[len(self.data) - 3].astype('float32')
        state[(8400 * 3):] = self.data[len(self.data) - 4].astype('float32')
        
        #divide by 256 to make values < 1 for neural net
        state /= 256.0
        
        #create theano variables
        state = theano.shared(state, name='input', borrow=True)
        state = T.reshape(state, (1, 80, 105, 4))
        state = T.cast(state, dtype='floatX')
        return state
        
    def reset_data(self):
        self.data = []
        self.actions = []
        self.rewards = []
        
