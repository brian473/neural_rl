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
                    gamma = .9, learning_rate = .5, momentum_step_size=.9):
        """
        Sets up variables for later use.
        
        """
        self.print_cost = True
        self.mini_batch_size = mini_batch_size
        self.num_mini_batches = num_mini_batches
        self.learning_rate = theano.shared(np.cast['float32'](learning_rate))
        self.history_size = history_size
        self.gamma = gamma
        self.data = []
        self.actions = []
        self.rewards = []
        self.terminal = []
        self.counter = 0
        self.momentum_step_size = momentum_step_size
        self.action = 0
        self.cnn = cnn
        self.image_shape = (4, 80, 80, self.mini_batch_size * self.num_mini_batches)
        
        self.rms_vals = []
    
        self.rms_vals.append(theano.shared(np.ones((4, 8, 8, 16), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((16, 19, 19), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((16, 4, 4, 32), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((32, 9, 9), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((2592, 256), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((256,), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((256, 3), dtype='float32')))
        self.rms_vals.append(theano.shared(np.ones((3,), dtype='float32')))
        
        self.momentum_vals = []
    
        self.momentum_vals.append(theano.shared(np.zeros((4, 8, 8, 16), dtype='float32')))
        self.momentum_vals.append(theano.shared(np.zeros((16, 19, 19), dtype='float32')))
        self.momentum_vals.append(theano.shared(np.zeros((16, 4, 4, 32), dtype='float32')))
        self.momentum_vals.append(theano.shared(np.zeros((32, 9, 9), dtype='float32')))
        self.momentum_vals.append(theano.shared(np.zeros((2592, 256), dtype='float32')))
        self.momentum_vals.append(theano.shared(np.zeros((256,), dtype='float32')))
        self.momentum_vals.append(theano.shared(np.zeros((256, 3), dtype='float32')))
        self.momentum_vals.append(theano.shared(np.zeros((3,), dtype='float32')))
        
        self.grad_vals = []

        self.grad_vals.append(theano.shared(np.zeros((4, 8, 8, 16), dtype='float32')))
        self.grad_vals.append(theano.shared(np.zeros((16, 19, 19), dtype='float32')))
        self.grad_vals.append(theano.shared(np.zeros((16, 4, 4, 32), dtype='float32')))
        self.grad_vals.append(theano.shared(np.zeros((32, 9, 9), dtype='float32')))
        self.grad_vals.append(theano.shared(np.zeros((2592, 256), dtype='float32')))
        self.grad_vals.append(theano.shared(np.zeros((256,), dtype='float32')))
        self.grad_vals.append(theano.shared(np.zeros((256, 3), dtype='float32')))
        self.grad_vals.append(theano.shared(np.zeros((3,), dtype='float32')))

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
        
        #theano_args contains information about the shape of each layer
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name,
                        batch_size=training_batch_size).astype("float32")
            theano_args.append(arg)
        theano_args = tuple(theano_args)
        
        y_hat = self.cnn.fprop(theano_args[0])
        
        #function used for faster fprop
        self.fprop_func = theano.function([theano_args[0]], y_hat)
        
        cost = self.cnn.cost(theano_args[1], y_hat)
       
        #params is the list of layers in the NN
        params = list(self.cnn.get_params())

        grads = T.grad(cost, params, disconnected_inputs='ignore')

        gradients = OrderedDict(izip(params, grads))
        
        rms_vals_dict = OrderedDict(izip(params, self.rms_vals))

        momentum_vals_dict = OrderedDict(izip(params, self.momentum_vals))
        
        grad_vals_dict = OrderedDict(izip(params, self.grad_vals))

        grad_update = OrderedDict()

        grad_update.update(dict(safe_zip(self.grad_vals, [gradients[param]
                                                    for param in params])))


        #function used for getting gradients
        #this is so that we only calculate gradients once, then
        #the same values are used for updating momentum, rmsprop, and training
        self.grad_update_func = theano.function(theano_args, updates=grad_update,
                                                on_unused_input='ignore')

        updates = OrderedDict()
        
        updates.update(dict(safe_zip(params, [param - self.learning_rate * 
                                (grad_vals_dict[param] / 
                                T.sqrt(rms_vals_dict[param] + 1e-8)) +
                                (self.momentum_step_size * 
                                momentum_vals_dict[param])
                                        for param in params])))
                                                    
        rmsprop_updates = OrderedDict()
        
        #rmsprop update function
        rmsprop_updates.update(dict(safe_zip(self.rms_vals, [(rms_vals_dict[param] * .9) + 
                                            (T.sqr(grad_vals_dict[param]) * .1)
                                                for param in params])))
        
        self.training = theano.function([], updates=updates, 
                                        on_unused_input='ignore')
                                        
        self.rmsprop_update = theano.function([], updates=rmsprop_updates,
                                                on_unused_input='ignore')
        
        momentum_updates = OrderedDict()

        #momentum update function
        momentum_updates.update(dict(safe_zip(self.momentum_vals, [-self.learning_rate * 
                                            (grad_vals_dict[param] / T.sqrt(rms_vals_dict[param] + 
                                                            1e-8)) + (self.momentum_step_size * 
                                                                momentum_vals_dict[param])
                                                                    for param in params])))

        self.momentum_update = theano.function([], updates=momentum_updates, 
                                                            on_unused_input='ignore')


        temp = T.tensor4()
        
        #function used for shuffling dimensions into c01b format
        self.dimshuf_func = theano.function([temp], temp.dimshuffle(1, 2, 3, 0))
        

        #functions to get grads and costs for debugging
        self.grads_func = theano.function(theano_args, grads)
        self.cost_function = theano.function(theano_args, cost)
        

    def add(self, data, action, reward, terminal):
        """
        Used for adding data to the dataset.

        """

        #make image square, must be square for the NN implementation
        image = Image.fromarray(data.reshape((80, 105)))
        image = image.resize((80, 80))
        
        image = np.asarray(image)
        
        self.data.append(image.astype('uint8'))
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminal.append(terminal)


        #remove first data point if history size is too large
        if len(self.data) > self.history_size:
            self.data.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.terminal.pop(0)


    def remove(self, num):
    # removes num last stored data points
        self.data = self.data[:len(self.data) - num - 1]
        self.actions = self.actions[:len(self.actions) - num - 1]
        self.rewards = self.rewards[:len(self.rewards) - num - 1]
        self.terminal = self.terminal[:len(self.terminal) - num - 1]
    
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
            #select a random point in history
            data_num = self.randGenerator.randint(8, len(self.data) - 5)
                
            for i in range(3):
                if self.terminal[data_num - (i + 1)]:
                    data_num -= (i + 1)
                    break

            #put values into lists
            if states == None:
                states = self.get_state(data_num)
                next_states = self.get_state(data_num + 1)
            else:
                states = np.append(states, self.get_state(data_num))
                next_states = np.append(next_states, self.get_state(data_num + 4))
            actions.append(self.actions[data_num])
            terminals.append(self.terminal[data_num])
            
            reward = self.rewards[data_num]
            #reward = 0

            #for i in range (4):
            #    reward += self.rewards[data_num - i + 4]

            rewards.append(reward)
        
        #normalize values
        states = states.reshape((32, 4, 80, 80)) / 128.0
        next_states = next_states.reshape((32, 4, 80, 80)) / 128.0
        
        states = self.dimshuf_func(states)
        next_states = self.dimshuf_func(next_states)
        
        #get Q(s, a) for state batch
        q_sa_list = self.fprop_func(states)
        
        #get Q(s, a)' for next state batch
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
        
        cost_before = self.cost_function(states, q_sa_list)


        #for i in self.grads_func(states, q_sa_list):
        #    print i.shape
                
        self.grad_update_func(states, q_sa_list)
        self.training()
        self.momentum_update()
        self.rmsprop_update()
        
        cost_after = self.cost_function(states, q_sa_list)

        print "Cost difference:", cost_before - cost_after

        #if (cost_after / cost_before < 1.0000001):
         #   self.learning_rate.set_value(np.cast['float32'](self.learning_rate.get_value() * 1.001))
        #else:
        #    self.learning_rate.set_value(np.cast['float32'](self.learning_rate.get_value() * .9))

    def get_state(self, num):
        #combines four images starting at num and going backwards
        #returns state as a 1D nparray
            
        state = np.append(self.data[num - 3].reshape((6400)).astype('float32'), self.data[num - 2].reshape((6400)).astype('float32'))
        state = np.append(state, self.data[num - 1].reshape((6400)).astype('float32'))
        state = np.append(state, self.data[num].reshape((6400)).astype('float32'))
        
        return state
