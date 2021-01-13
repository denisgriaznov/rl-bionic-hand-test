# Using Deep Reinforcement Learning to Simulate Bionic Arm Control

This is a small research aimed at testing the reinforcement learning capabilities for autonomous control of the bionic arm.  The main goal in the future is to achieve full autonomy in the control of the bionic arm, so that it adapts to anytask in the environment.

You can see more detailed description of research at [Description pdf](description.pdf) 

To implement reinforcement learning, the Gym library from OpenAI was chosen.  

It worked inconjunction with Unity ML Agents via Python. A model of a hand with 15 degrees of freedom was built.  The game consisted in grabbing an item that appeared in one of 3 different positions at random.

![test](/Images/gif.gif)
