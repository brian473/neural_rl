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
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

from random import Random

import numpy as np

import matplotlib.pyplot as plt
import time

class RandomAgent(Agent):
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
        self.saving = True

        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(task_spec_string)
        if TaskSpec.valid:
            
            assert ((len(TaskSpec.getIntObservations())== 0) !=
                    (len(TaskSpec.getDoubleObservations()) == 0 )), \
                "expecting continous or discrete observations.  Not both."
            assert len(TaskSpec.getDoubleActions())==0, \
                "expecting no continuous actions"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), \
                " expecting min action to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), \
                " expecting max action to be a number not a special value"
            self.num_actions=TaskSpec.getIntActions()[0][1]+1;

        
        self.int_states = len(TaskSpec.getIntObservations()) > 0

        # Create empty lists for data collection. 
        self.states = []
        self.actions = []
        self.rewards = []
        self.absorbs = []

        #Create appropriate RL-Glue objects for storing these. 
        self.last_action=Action()
        self.last_observation=Observation()

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

        #Generate random action, 0 or 1
        self.start_time = time.time()
        this_int_action=self.randGenerator.randint(0,self.num_actions-1)
        return_action=Action()
        return_action.intArray=[this_int_action]


        self.last_action=copy.deepcopy(return_action)
        self.last_observation=copy.deepcopy(observation)

        return return_action


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
            self.image = plt.imshow(img/256.0, interpolation='none',cmap="gray")
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
        # Generate random action
        this_int_action=self.randGenerator.randint(0,self.num_actions-1)
        return_action=Action()
        return_action.intArray=[this_int_action]
        
        if self.show_ale:
            #self._show_ale_color()
            self._show_ale_gray()

        if self.saving:
            if self.int_states:
                self.states.append(self.last_observation.intArray)
            else:
                self.states.append(self.last_observation.doubleArray)

            self.actions.append(self.last_action.intArray[0])
            self.rewards.append(reward)
            self.absorbs.append(False)

        self.last_action=copy.deepcopy(return_action)
        self.last_observation=copy.deepcopy(observation)

        return return_action

    def agent_end(self, reward):
        """
        This function is called once at the end of an episode. 
        
        Arguments: 
           reward      - Real valued reward.

        Returns: 
            None
        """
        if self.int_states:
            self.states.append(self.last_observation.intArray)
            # JUST REPEAT THE PREVIOUS OBSERVATION.  THIS SHOULD
            # BE IGNORED ANYWAY.
        else:
            self.states.append(self.last_observation.doubleArray)

        self.actions.append(self.last_action.intArray[0])
        self.rewards.append(reward)
        self.absorbs.append(True)

    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """
        pass

    def agent_message(self, in_message):
        """
        The experiment will cause this method to be called.  Used
        to save data to the indicated file. 
        """
        if in_message.startswith("save_data"):
            total_time = time.time() - self.start_time
            file_name=in_message.split(" ")[1]
            the_file = open(file_name, "w")
            all_data = (np.array(self.states), 
                        np.array(self.actions), 
                        np.array(self.rewards),
                        np.array(self.absorbs))
            print "PICKLING: " + file_name
            cPickle.dump(all_data, the_file, -1)
            print "Simulated at a rate of {}/s".format(len(self.rewards) / 
                                                       total_time)
            return "File saved successfully";

        else:
            return "I don't know how to respond to your message";


if __name__=="__main__":
    AgentLoader.loadAgent(RandomAgent())
