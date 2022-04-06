# MRS_DQN_2D

A Reinforcement learning based Deep Q Network controller for Multi Robot System to transport an object from starting position to a goal location in a 2D setup. This implementation takes the relative distance between robot and goal, and the angle as the inputs to the controller and achieves the goal with precision. 

The kinmeatics of the system is formulated from the reference publication included in the repository. 

dqn_train.py contains the Deep Q-netwrok Reinforcement learning training code. The environment is formulated in the environment.py file. 
To visualize the simulation, please run the run_simulation.py file.
