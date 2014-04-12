from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from theano import ProfileMode
import numpy as np
import theano
from theano.compat.python2x import OrderedDict
from pylearn2.space import VectorSpace
from pylearn2.utils import safe_zip
import theano.tensor as T
import copy
import os
from itertools import izip
from pylearn2.utils.data_specs import DataSpecsMapping
import Image
import matplotlib.pyplot as plt
import time

from random import Random

class NeuralQLearnDataset:
    randGenerator=Random()


    def __init__(self, cnn, mini_batch_size, num_mini_batches, history_size = 100000, \
                    gamma = .9, learning_rate = .5):
        """
        Sets up variables for later use.
        
        """
        self.print_cost = False
        self.mini_batch_size = mini_batch_size
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate
        self.history_size = history_size
        self.gamma = gamma
        self.data = []
        self.actions = []
        self.rewards = []
        self.terminal = []
        self.counter = 0
        self.action = 0
        self.cnn = cnn
        self.image_shape = (4, 80, 80, self.mini_batch_size * self.num_mini_batches)
        
        self.setup_training()
        
    def setup_training(self):
        """
        Sets up training function.
        """
        
        training_batch_size = self.mini_batch_size
        
        cost = self.cnn.get_default_cost()
        
        data_specs = cost.get_data_specs(self.cnn)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
        
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name,
                        batch_size=training_batch_size).astype("float32")
            theano_args.append(arg)
        theano_args = tuple(theano_args)
        
        y_hat = self.cnn.fprop(theano_args[0])
        
        self.fprop_func = theano.function([theano_args[0]], y_hat)
        
        cost = self.cnn.cost(theano_args[1], y_hat)
        
        lr_scalers = self.cnn.get_lr_scalers()
        
        params = list(self.cnn.get_params())
        grads = T.grad(cost, params, disconnected_inputs='ignore')
        
        gradients = OrderedDict(izip(params, grads))
        
        updates = OrderedDict()
        
        updates.update(dict(safe_zip(params, [param - self.learning_rate * 
                                gradients[param] * lr_scalers.get(param, 1.) for param in params])))
        
        self.training = theano.function(theano_args, updates=updates, 
                                        on_unused_input='ignore')
                                        
        self.cost_function = theano.function(theano_args, cost)
        

    def add(self, data, action, reward, terminal):
        """
        Used for adding data to the dataset.

        """
        image = Image.fromarray(data.reshape((80, 105)))
        image = image.resize((80, 80))
        
        image = np.asarray(image)
        
        self.data.append(image.astype('uint8'))
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminal.append(terminal)

        if len(self.data) > self.history_size:
            self.data.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.terminal.pop(0)

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
        states = state = np.empty((self.mini_batch_size * 
                        self.num_mini_batches, 25600), dtype='float32')
        next_states = state = np.empty((self.mini_batch_size * 
                        self.num_mini_batches, 25600), dtype='float32')
        actions = []
        rewards = []
        terminal_state = False
        
        #create training batch
        for i in range(self.mini_batch_size * self.num_mini_batches):
            #select a random point in history
            data_num = self.randGenerator.randint(3, len(self.data) - 5)
            
            for i in range(0, 8):
                if self.terminal[i - 3 + data_num]:
                    terminal_state = True
                    data_num -= 7 - i
                    
            
            #put values into lists
            states[i] = self.get_state(data_num)
            next_states[i] = self.get_state(data_num + 4)
            actions.append(self.actions[data_num])
            
            reward = 0
            for i in range(4):
                if self.rewards[i + 1 + data_num] != 0:
                    reward = self.rewards[i + 1 + data_num]
            
            rewards.append(reward)
        
        #normalize values
        states /= 256.0
        next_states /= 256.0
        
        #get output predictions from nn using the states batch
        q_sa_list = self.fprop_func(states.reshape(self.image_shape))
        
        #get max of Q(s, a)' for next state batch
        q_sa_prime_list = self.fprop_func(next_states.reshape(self.image_shape))
        
        print np.mean(q_sa_list)
        if not terminal_state:
            for i in range(self.mini_batch_size * self.num_mini_batches):
                #perform qlearning update to get target value Q(s, a)
                next_state_max = np.max(q_sa_prime_list[i])
                q_sa_list[i][actions[i]] = rewards[i] + (self.gamma * next_state_max)
        else:
            for i in range(self.mini_batch_size * self.num_mini_batches):
                #in terminal state, target value is reward
                q_sa_list[i][actions[i]] = rewards[i]
        
        if self.print_cost:
            print self.cost_function(states_np_array.reshape(self.image_shape), 
                                                              q_sa_list)

        #perform SGD update with minibatches
        for i in range(self.num_mini_batches):
            batch_q_sa = q_sa_list[i * self.mini_batch_size : 
                                        (1 + i) * 
                                        self.mini_batch_size]
            batch_data = states[i * self.mini_batch_size : 
                                            (1 + i) * 
                                            self.mini_batch_size]
                                                
            batch_data = batch_data.reshape((4, 80, 80, self.mini_batch_size))

            self.training(batch_data, batch_q_sa)
        
        if self.print_cost:
            print self.cost_function(states.reshape(self.image_shape), 
                                                            q_sa_list)
        

    def get_state(self, num):
        #combine four states at num
        state = np.empty(25600)
            
        state[:6400] = self.data[num - 3].reshape(6400).astype('float32')
        state[6400:(6400 * 2)] = self.data[num - 2].reshape(6400).astype('float32')
        state[(6400 * 2):(6400 * 3)] = self.data[num - 1].reshape(6400).astype('float32')
        state[(6400 * 3):] = self.data[num].reshape(6400).astype('float32')
        
        return state

    def get_cur_state(self):
        #combine last four states
        state = self.get_state(len(self.data) - 1)
        
        #divide by 256 to make values < 1 for neural net
        state /= 256.0
        
        #create theano variables
        state = theano.shared(state, name='input')
        state = T.reshape(state, (4, 80, 80, 1))
        state = T.cast(state, dtype='floatX')
        return state
        
    def reset_data(self):
        self.data = []
        self.actions = []
        self.rewards = []
        
