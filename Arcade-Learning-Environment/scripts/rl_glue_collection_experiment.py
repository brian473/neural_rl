#!/usr/bin/env python
""" 
This is an Rl-Glue experiment script designed to collect
a set of randomly collected experience tuples suitable for batch
reinforcement learning. 

Based on the sample_experiment.py from the Rl-glue python codec examples.

Author: Nathan Sprague
"""


import rlglue.RLGlue as RLGlue
import time

def main():
    num_episodes = 2
    max_steps_per_episode = 18000
    RLGlue.RL_init()
    for episode in range(0,num_episodes):
        RLGlue.RL_episode(max_steps_per_episode)
        print "Episode finished.", time.time()
    
    RLGlue.RL_agent_message("save_data data.pkl");


if __name__ == "__main__":
    main()
