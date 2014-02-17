=================================
RL-Glue Python MINES-SARSA-SAMPLE README
=================================
----------------------------
Introduction
----------------------------
This is a sample experiment that has the "Mines" environment and a simple tabular Sarsa agent.  This project lives in RL-Library, but is also distributed with the RL-Glue Python Codec.

This example requires the Python Codec:
http://glue.rl-community.org/Home/Extensions/python-codec

Running
----------------------------
- These instructions assume that you have rl_glue (or rl_glue.exe) installed on your path so that you don't have to type the full path to it.
- They also assume that the RL-Glue Python codec has been installed to your Python path.  If not, you will need to set your Python path to include them or add it at each step (one example is given below).

Do the following in different console/terminal windows:
#If you want to do them in the same terminal window, append an ampersand & to each line
$> python sample_sarsa_agent.py
#Alternatively, if you don't have the Python codec on your Python path
#$> PYTHONPATH=/path/to/python/codec/src python sample_sarsa_agent.py

$> python sample_mines_environment.py
$> python sample_experiment.py
$> rl_glue #(maybe rl_glue.exe)


----------------------------
More Information
----------------------------
Please see the Python Codec Manual and FAQ if you are looking for more information:
http://glue.rl-community.org/Home/Extensions/python-codec


-- 
Brian Tanner
btanner@rl-community.org

