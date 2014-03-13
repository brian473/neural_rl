from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
import numpy as np
import theano
from theano.compat.python2x import OrderedDict
from pylearn2.space import VectorSpace
from pylearn2.utils import safe_zip
import theano.tensor as T
import copy
from itertools import izip
from pylearn2.utils.data_specs import DataSpecsMapping

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
        self.train_mini_batches = True
        
        self.setup_training()
        
    def setup_training(self):
        """
        Sets up training function.
        """
        
        training_batch_size = self.batch_size
        if self.train_mini_batches:
            training_batch_size = 32
        
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
        
        cost = self.cnn.cost(theano_args[1], y_hat)
        
        params = list(self.cnn.get_params())
        grads = T.grad(cost, params, disconnected_inputs='ignore')
        
        gradients = OrderedDict(izip(params, grads))
        
        updates = OrderedDict()
        
        updates.update(dict(safe_zip(params, [param - self.learning_rate * 
                                gradients[param] for param in params])))
                                
        self.training = theano.function(theano_args, updates=updates, 
                                        on_unused_input='ignore',)
        

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
        states = state = np.empty((self.batch_size, 33600), dtype='float32')
        next_states = state = np.empty((self.batch_size, 33600), dtype='float32')
        actions = []
        rewards = []
        
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
            states[i] = state
            next_states[i] = state
            actions.append(self.actions[data_num])
            rewards.append(self.rewards[data_num])
        
        #normalize values
        states /= 256.0
        next_states /= 256.0
        
        states_np_array = states
        
        #create theano tensors
        states = theano.shared(states, name='input')
        states = T.reshape(states, (self.batch_size, 80, 105, 4))
        states = T.cast(states, dtype='floatX')
        
        next_states = theano.shared(next_states, name='input_max')
        next_states = T.reshape(next_states, (self.batch_size, 80, 105, 4))
        next_states = T.cast(next_states, dtype='floatX')
        
        #get output predictions from nn using the states batch
        q_sa_list = self.cnn.fprop(states).eval()
        
        #get max of Q(s, a)' for next state batch
        q_sa_prime_list = self.cnn.fprop(next_states).eval()
        
        for i in range(self.batch_size):
            #perform qlearning update to get target value Q(s, a)
            next_state_max = max(q_sa_prime_list[i])
            q_sa_list[i][actions[i]] = q_sa_list[i][actions[i]] + (self.alpha * \
                            (rewards[i] + (self.gamma * next_state_max) - \
                            q_sa_list[i][actions[i]]))
            
        #perform SGD update
        num_mini_batches = self.batch_size / 32
        
        if self.train_mini_batches:
            for i in range(num_mini_batches - 1):
                batch_q_sa = q_sa_list[i * num_mini_batches : 32 + i * \
                                                    num_mini_batches]
                batch_data = states_np_array[i * num_mini_batches : 32 + i * \
                                                    num_mini_batches]
                                                    
                batch_data = batch_data.reshape((32, 80, 105, 4))

                self.training(batch_data, batch_q_sa)
        else:
            states_np_array = states_np_array.reshape((self.batch_size, 80, 105, 4))
            self.training(states_np_array, q_sa_list)
        
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
        state = theano.shared(state, name='input')
        state = T.reshape(state, (1, 80, 105, 4))
        state = T.cast(state, dtype='floatX')
        return state
        
    def reset_data(self):
        self.data = []
        self.actions = []
        self.rewards = []
        
