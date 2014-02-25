from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
import numpy as np
import theano
import theano.tensor as T

from random import Random


class NeuralQLearnDataset:

    randGenerator=Random()


    def __init__(self, cnn, history_size = 100000, gamma = .9, alpha = .1, batch_size = 32):
        """
        Sets up variables for later use.
        
        """
        self.batch_size = batch_size
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
        The four channels contain subsequent states. Performs QLearning update step for the target, using 
        the most recent state's action and reward values.

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
        
        
        #create training batch of size 256
        for i in range(self.batch_size):
            #select a random point in history
            data_num = self.randGenerator.randint(3, len(self.data) - 2)
            
            #combine four states for training
            state = np.append(self.data[data_num].astype('float32'), self.data[data_num - 1].astype('float32'), axis=0)
            state = np.append(state, self.data[data_num - 2].astype('float32'), axis=0)
            state = np.append(state, self.data[data_num - 3].astype('float32'), axis=0)
            
            #combine the next four states for Q(s, a)'
            next_state = np.append(self.data[data_num + 1].astype('float32'), self.data[data_num].astype('float32'), axis=0)
            next_state = np.append(next_state, self.data[data_num - 1].astype('float32'), axis=0)
            next_state = np.append(next_state, self.data[data_num - 2].astype('float32'), axis=0)
            
            #put values into lists
            states.append(state)
            next_states.append(next_state)
            actions.append(self.actions[data_num])
            rewards.append(self.rewards[data_num])
            
            #dataset requires data to be one dimension
            states_nparray.append(state.reshape(80 * 105 * 4).astype(dtype='float32'))
     
        #create theano variables
        states = np.asarray(states)
        states /= 256.0
        states = theano.shared(states, name='input')
        states = T.reshape(states, (self.batch_size, 80, 105, 4))
        states = T.cast(states, dtype='floatX')
            
        next_states = np.asarray(next_states)
        next_states /= 256.0
        next_states = theano.shared(next_states, name='input_max')
        next_states = T.reshape(next_states, (self.batch_size, 80, 105, 4))
        next_states = T.cast(next_states, dtype='floatX')
            
        #get output predictions from nn using the states batch
        q_sa_list = self.cnn.fprop(states).eval()
            
        #get max of Q(s, a)' for next state batch
        q_sa_prime_list = self.cnn.fprop(next_states).eval()
        
        states_nparray = np.asarray(states_nparray)
        states_nparray /= 256.0
        
        for i in range(self.batch_size):
            #perform qlearning update to get target value Q(s, a)
            next_state_max = max(q_sa_prime_list[i])
            q_sa_list[i][actions[i]] = q_sa_list[i][actions[i]] + (self.alpha * (rewards[i] + (self.gamma * next_state_max) - q_sa_list[i][actions[i]]))
            
        view_converter = DefaultViewConverter((80, 105, 4), ('b', 0, 1, 'c'))
        return DenseDesignMatrix(X=np.asarray(states_nparray).astype('float32'), y=np.asarray(q_sa_list).astype('float32'), view_converter=view_converter)
        
    def get_cur_state(self):
        #combine four states
        data = np.append(self.data[len(self.data) - 1].astype('float32'), self.data[len(self.data) - 2].astype('float32'), axis=0)
        data = np.append(data, self.data[len(self.data) - 3].astype('float32'), axis=0)
        data = np.append(data, self.data[len(self.data) - 4].astype('float32'), axis=0)
        
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
        
