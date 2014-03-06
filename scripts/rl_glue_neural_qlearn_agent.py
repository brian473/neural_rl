#!/usr/bin/env python
"""
This agent interacts with an RL-Glue environment to collect
a set of (s, a, r, absorb) experiences for batch reinforceement learning. 

This uses the skeleton_agent.py file from the Python-codec of rl-glue
as a starting point. 

Author: Nathan Sprague
"""


# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import copy
import cPickle
import theano
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from pylearn2.training_algorithms import sgd
from pylearn2.models import mlp
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import EpochCounter
import neural_qlearn_dataset as nqd
import time
import theano.tensor as T

from random import Random

import numpy as np

import matplotlib.pyplot as plt
import time

class NeuralQLearnAgent(Agent):
    randGenerator=Random()

    def agent_init(self,task_spec_string):
        """ 
        This function is called once at the beginning of an experiment.

        Arguments: task_spec_string - A string defining the task.  This string
                                      is decoded using 
                                      TaskSpecVRLGLUE3.TaskSpecParser
        """

        self.image = None
        self.show_ale = False
        self.saving = False
        self.total_reward = 0
        self.batch_size = 1024 #must be a multiple of 32
        self.episode_count = 0
        learning_rate = 10
        self.testing_policy = False
        load_file = False
        load_file_name = "cnnparams.pkl"
        self.save_file_name = "cnnparams.pkl"
        
        #starting value for epsilon-greedy
        self.epsilon = 1

        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(task_spec_string)
        if TaskSpec.valid:
            
            assert ((len(TaskSpec.getIntObservations())== 0) != \
                (len(TaskSpec.getDoubleObservations()) == 0 )), \
                "expecting continous or discrete observations.  Not both."
            assert len(TaskSpec.getDoubleActions())==0, \
                "expecting no continuous actions"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), \
                " expecting min action to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), \
                " expecting max action to be a number not a special value"
            self.num_actions=TaskSpec.getIntActions()[0][1]+1

        
        self.int_states = len(TaskSpec.getIntObservations()) > 0

        # Create neural network and initialize trainer and dataset
        
        if load_file:
            thefile = open(load_file_name, "r")
            
            self.cnn = cPickle.load(thefile)
        else:
        
            self.first_conv_layer = mlp.ConvRectifiedLinear(16, (8, 8), (1, 1), 
                            (1, 1), "first conv layer", irange=.1, 
                                            kernel_stride=(4, 4))
                                            
            self.second_conv_layer = mlp.ConvRectifiedLinear(32, (4, 4), 
                            (1, 1), (1, 1), "second conv layer", irange=.1, 
                                            kernel_stride=(2, 2))
                                            
            self.rect_layer = mlp.RectifiedLinear(dim=256, 
                            layer_name="rectified layer", irange=.1)
                            
            self.output_layer = mlp.Linear(self.num_actions, "output layer", 
                            irange=.1)

            layers = [self.first_conv_layer, self.second_conv_layer, 
                            self.rect_layer, self.output_layer]

            self.cnn = mlp.MLP(layers, input_space = Conv2DSpace((80, 105), 
                            num_channels=4), batch_size=self.batch_size)

        self.data = nqd.NeuralQLearnDataset(self.cnn, batch_size = 
                        self.batch_size, learning_rate=learning_rate)
                        

        #Create appropriate RL-Glue objects for storing these. 
        self.last_action=Action()
        self.last_observation=Observation()
        
        print "Policy test results"



    def agent_start(self,observation):
        """  
        This method is called once at the beginning of each episode. 
        No reward is provided, because reward is only available after
        an action has been taken. 

        Arguments: 
           observation - An observation of type rlglue.types.Observation

        Returns: 
           An action of type rlglue.types.Action
        """

        self.start_time = time.time()
        this_int_action = self.get_action()
        return_action=Action()
        return_action.intArray = [this_int_action]


        self.last_action = copy.deepcopy(return_action)
        self.last_observation = copy.deepcopy(observation)

        return return_action
        
    def get_action(self):
        """
        Get predictions on the last state to get the next action from qvalues.
        Uses epsilon-greedy.
        """
        #reduce epsilon linearly to .1 after 900,000 steps
        epsilon = self.epsilon
        
        if self.testing_policy:
            epsilon = .01
        else:
            self.epsilon -= .000001
            if (self.epsilon < .1):
                self.epsilon = .1
                
        if self.randGenerator.random() < epsilon or len(self.data) <= 7:
            return self.randGenerator.randint(0,self.num_actions-1)
            
        state = self.data.get_cur_state()
        
        qvalues = self.cnn.fprop(state).eval()
        
        qvalues = qvalues.tolist()
        
        return qvalues[0].index(max(qvalues[0]))

    def _show_ale_color(self):
        """
        Show an arcade learning environment screen.
        """
        plt.ion()
        img = np.array(\
            self.last_observation.intArray).reshape((160, 210, 3))
        img = np.transpose(img, (1,0,2))
        if self.image == None:
            self.image = plt.imshow(img/256.0, interpolation='none')
            fig = plt.gcf()
            fig.set_size_inches(4,3)
            plt.show()
        else:
            self.image.set_data(img/256.0) 
        plt.draw()
        time.sleep(.0)

    def _show_ale_gray(self):
        """
        Show an arcade learning environment screen.
        """
        plt.ion()
        img = np.array(\
            self.last_observation.intArray).reshape((80, 105))
        img = np.transpose(img)
        if self.image == None:
            self.image = plt.imshow(img/256.0, interpolation='none', \
                                            cmap="gray")
            fig = plt.gcf()
            fig.set_size_inches(4,3)
            plt.show()
        else:
            self.image.set_data(img/256.0)
        plt.draw()
        time.sleep(.0)

    def agent_step(self, reward, observation):
        """
        This method is called each time step. 

        Arguments: 
           reward      - Real valued reward.
           observation - An observation of type rlglue.types.Observation

        Returns: 
           An action of type rlglue.types.Action
        
        """
        
        self.total_reward += reward
        
        #set reward to be either -1, 0, or 1 as described in atari paper
        if reward > 0:
            reward = 1
		
        if reward < 0:
            reward = -1
        
        self.data.add(self.last_observation.intArray, \
                        self.last_action.intArray[0], reward)
        
        if len(self.data) > 1000 and not self.testing_policy:
            t = time.time()
            self.data.train()
            print "took ", time.time() - t, "s"
        
        this_int_action = self.get_action()
        return_action = Action()
        return_action.intArray = [this_int_action]
        
        if self.show_ale:
            #self._show_ale_color()
            self._show_ale_gray()

        self.last_action = copy.deepcopy(return_action)
        self.last_observation = copy.deepcopy(observation)

        return return_action

    def agent_end(self, reward):
        """
        This function is called once at the end of an episode. 
        
        Arguments: 
           reward      - Real valued reward.

        Returns: 
            None
        """
        self.total_reward += reward
        
        if self.testing_policy:
            print self.total_reward
        
        self.total_reward = 0
        if len(self.data) > 1000:
            if self.episode_count == 1 and not self.testing_policy:
                self.testing_policy = True
            elif self.testing_policy:
                self.episode_count = 0
                self.testing_policy = False
                if self.saving:
                    self.save_params(self.save_file_name)
            else:
                self.episode_count+=1
            
        
        if reward > 0:
            reward = 1
		
        if reward < 0:
            reward = -1
        
        self.data.add(self.last_observation.intArray, \
                        self.last_action.intArray[0], reward)
        
        if len(self.data) > 10000:
            self.data.train()
            
        #self.data.reset_data()
        

    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """
        pass
        
    def save_params(self, filename="cnnparams.pkl"):
        the_file = open(filename, "w")
        
        cPickle.dump(self.cnn, the_file, -1)

    def agent_message(self, in_message):
        """
        The experiment will cause this method to be called.  Used
        to save data to the indicated file. 
        """
        if in_message.startswith("save_data"):
            total_time = time.time() - self.start_time
            file_name=in_message.split(" ")[1]
            the_file = open(file_name, "w")
            all_data = (self.cnn.get_params())
            print "PICKLING: " + file_name
            cPickle.dump(all_data, the_file, -1)
            #print "Simulated at a rate of {}/s".format(len(self.rewards) / 
            #                                           total_time)
            return "File saved successfully"

        else:
            return "I don't know how to respond to your message"


if __name__=="__main__":
    AgentLoader.loadAgent(NeuralQLearnAgent())
