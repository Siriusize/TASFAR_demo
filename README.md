# Demo code for TASFAR
### Introduction 
In this repo, we demonstrate TASFAR by pedestrian dead reckoning (location sensing), as in one of the experiments from the original paper. In pedestrian dead reckoning, a model utilizes the inertial measurement unit (IMU) signals to estimate the 2D walking trajectory of phone users. 

In the experiment, we use MCNN [1] as baseline model and show TASFAR can further improves its performance on each user.

This demo consists of two parts:
- Pseudo-label generation of TASFAR
- Training on pseudo-labels
- Testing on adapted models

# Kick start
'''
abd
'''


# Reference
[1] Herath S, Yan H, Furukawa Y. Ronin: Robust neural inertial navigation in the wild: Benchmark, evaluations, & new methods[C]//2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020: 3146-3152.