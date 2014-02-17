# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
# 
# $Revision$
# $Date$
# $Author$
# $HeadURL$
#



import sys
import math
import rlglue.RLGlue as RLGlue

# TO USE THIS Experiment [order doesn't matter]
# NOTE: I'm assuming the Python codec is installed an is in your Python path
#   -  Start the rl_glue executable socket server on your computer
#   -  Run the SampleSarsaAgent and SampleMinesEnvironment from this or a
#   different codec (Matlab, Python, Java, C, Lisp should all be fine)
#   -  Start this environment like:
#   $> python sample_experiment.py

# Experiment program that does some of the things that might be important when
# running an experiment.  It runs an agent on the environment and periodically
# asks the agent to "freeze learning": to stop updating its policy for a number
# of episodes in order to get an estimate of the quality of what has been learned
# so far.
#
# The experiment estimates statistics such as the mean and standard deviation of
# the return gathered by the policy and writes those to a comma-separated value file
# called results.csv.
#
# This experiment also shows off some other features that can be achieved easily
# through the RL-Glue env/agent messaging system by freezing learning (described
# above), having the environment start in specific starting states, and saving
# and loading the agent's value function to/from a binary data file.




#
#	This function will freeze the agent's policy and test it after every 25 episodes.
#
def offlineDemo():
	statistics=[];
	this_score=evaluateAgent();
	printScore(0,this_score);
	statistics.append(this_score);
	
	for i in range(0,20):
		for j in range(0,25):
			RLGlue.RL_episode(0);
		this_score=evaluateAgent();
		printScore((i+1)*25,this_score);
		statistics.append(this_score);
	
	saveResultToCSV(statistics,"results.csv");

def printScore(afterEpisodes, score_tuple):
	print "%d\t\t%.2f\t\t%.2f" % (afterEpisodes, score_tuple[0], score_tuple[1])

#
# Tell the agent to stop learning, then execute n episodes with his current
# policy.  Estimate the mean and variance of the return over these episodes.
#
def evaluateAgent():
	sum=0;
	sum_of_squares=0;
	this_return=0;
	mean=0;
	variance=0;
	n=10;
	
	RLGlue.RL_agent_message("freeze learning");
	for i in range(0,n):
		# We use a cutoff here in case the 
		#policy is bad and will never end an episode
		RLGlue.RL_episode(5000);
		this_return=RLGlue.RL_return();
		sum+=this_return;
		sum_of_squares+=this_return**2;
	
	mean=sum/n;
	variance = (sum_of_squares - n*mean*mean)/(n - 1.0);
	standard_dev=math.sqrt(variance);

	RLGlue.RL_agent_message("unfreeze learning");
	return mean,standard_dev;


def saveResultToCSV(statistics, fileName):
	theFile = open(fileName, "w");
	theFile.write("#Results from sample_experiment.py.  First line is means, second line is standard deviations.\n");

	for thisEntry in statistics:
		theFile.write("%.2f, " % thisEntry[0])
	theFile.write("\n");

	for thisEntry in statistics:
		theFile.write("%.2f, " % thisEntry[1])
	theFile.write("\n");

	theFile.close();


#
# Just do a single evaluateAgent and print it
#
def	single_evaluation():
	this_score=evaluateAgent();
	printScore(0,this_score);



print "Starting offline demo\n----------------------------\nWill alternate learning for 25 episodes, then freeze policy and evaluate for 10 episodes.\n"
print "After Episode\tMean Return\tStandard Deviation\n-------------------------------------------------------------------------"
RLGlue.RL_init()
offlineDemo()

print "\nNow we will save the agent's learned value function to a file...."

RLGlue.RL_agent_message("save_policy results.dat");

print "\nCalling RL_cleanup and RL_init to clear the agent's memory..."

RLGlue.RL_cleanup();
RLGlue.RL_init();

print "Evaluating the agent's default policy:\n\t\tMean Return\tStandardDeviation\n------------------------------------------------------"
single_evaluation();

print "\nLoading up the value function we saved earlier."
RLGlue.RL_agent_message("load_policy results.dat");

print "Evaluating the agent after loading the value function:\n\t\tMean Return\tStandardDeviation\n------------------------------------------------------"
single_evaluation();

print "Telling the environment to use fixed start state of 2,3."
RLGlue.RL_env_message("set-start-state 2 3");
RLGlue.RL_start();
print "Telling the environment to print the current state to the screen."
RLGlue.RL_env_message("print-state");
print "Evaluating the agent a few times from a fixed start state of 2,3:\n\t\tMean Return\tStandardDeviation\n-------------------------------------------"
single_evaluation();

print "Evaluating the agent again with the random start state:\n\t\tMean Return\tStandardDeviation\n-----------------------------------------------------";
RLGlue.RL_env_message("set-random-start-state");
single_evaluation();


RLGlue.RL_cleanup();
print "\nProgram Complete."




