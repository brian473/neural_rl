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


    def __init__(self, cnn, mini_batch_size, num_mini_batches, history_size = 1000000, \
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
        
        self.rms_vals = []
        
        self.rms_vals.append(theano.shared(np.ones((4, 8, 8, 64), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((64, 19, 19), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((64, 4, 4, 128), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((128, 9, 9), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((10368, 256), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((256,), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((256, 3), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((3,), dtype='float32')))
        
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
        
        rms_vals_dict = OrderedDict(izip(params, self.rms_vals))
        
        updates = OrderedDict()
        
        updates.update(dict(safe_zip(params, [param - self.learning_rate * 
                                (gradients[param] / 
                                T.sqrt(rms_vals_dict[param] + 1e-8)) 
                                for param in params])))
                                                    
        rmsprop_updates = OrderedDict()
        
        rmsprop_updates.update(dict(safe_zip(self.rms_vals, [(rms_vals_dict[param] * .9) + 
                                            (T.sqr(gradients[param]) * .1)
                                                for param in params])))
        
        self.training = theano.function(theano_args, updates=updates, 
                                        on_unused_input='ignore')
                                        
        self.rmsprop_update = theano.function(theano_args, updates=rmsprop_updates,
                                                on_unused_input='ignore')
        
        temp = T.tensor4()
        
        self.dimshuf_func = theano.function([temp], temp.dimshuffle(1, 2, 3, 0))
        
        #self.grads_func = theano.function(theano_args, grads)
                                        
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
        formatted_data = []
        states = None
        next_states = None
        actions = []
        rewards = []
        terminals = []
        
        #create training batch
        for i in range(self.mini_batch_size * self.num_mini_batches):
            #select a random point in history, multiple of 4 so state matches
            #action set
            data_num = (self.randGenerator.randint(8, (len(self.data) - 5) / 4) * 4) - 2
            
            terminal_state = False
            
            #if the state is terminal, subtract 4
            for i in range(4):
                if self.terminal[data_num - i]:
                    terminal_state = True
                    data_num -= 4
                    
            #determine if the next state is terminal
            for i in range(4):
                if self.terminal[data_num + 4 - i]:
                    terminal_state = True
            
            #put values into lists
            if states == None:
                states = self.get_state(data_num)
                next_states = self.get_state(data_num + 4)
            else:
                states = np.append(states, self.get_state(data_num))
                next_states = np.append(next_states, self.get_state(data_num + 4))
            actions.append(self.actions[data_num])
            terminals.append(terminal_state)
            
            reward = 0
            for i in range(4):
                if self.rewards[data_num - i] != 0:
                    reward += self.rewards[data_num - i]
            
            rewards.append(reward)
        
        #normalize values
        states = states.reshape((32, 4, 80, 80)) / 256.0
        next_states = next_states.reshape((32, 4, 80, 80)) / 256.0
        
        states = self.dimshuf_func(states)
        next_states = self.dimshuf_func(next_states)
        
        #get output predictions from nn using the states batch
        q_sa_list = self.fprop_func(states)
        
        #get max of Q(s, a)' for next state batch
        q_sa_prime_list = self.fprop_func(next_states)
        
        print np.mean(q_sa_list)
        print q_sa_list[0]
        print q_sa_prime_list[0]
        #print ""
        
        #for param in self.cnn.get_params():
        #    value = param.get_value(borrow=True)
        #    if np.any(np.isnan(value)) or np.any(np.isinf(value)):
        #        raise Exception("NaN in " + param.name)
        
        
        for i in range(self.mini_batch_size * self.num_mini_batches):
            #perform qlearning update to get target value Q(s, a)
            next_state_max = np.max(q_sa_prime_list[i])
            if not terminals[i]:
                q_sa_list[i][actions[i]] = rewards[i] + (self.gamma * next_state_max)
            else:
                q_sa_list[i][actions[i]] = rewards[i]
        
        if self.print_cost:
            print self.cost_function(states, q_sa_list)

        self.training(states, q_sa_list)
        self.rmsprop_update(states, q_sa_list)
        
        if self.print_cost:
            print self.cost_function(states, q_sa_list)
        
        
        #for i in self.grads_func(states.reshape(self.image_shape), q_sa_list):
        #    print i.shape
        

    def get_state(self, num):
        #combine four states at num
            
        state = np.append(self.data[num - 3].reshape((6400)).astype('float32'), self.data[num - 2].reshape((6400)).astype('float32'), axis=0)
        state = np.append(state, self.data[num - 1].reshape((6400)).astype('float32'), axis=0)
        state = np.append(state, self.data[num].reshape((6400)).astype('float32'), axis=0)
        
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
        
