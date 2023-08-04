# Demo code for TASFAR
## Introduction 
In this repo, we demonstrate TASFAR by pedestrian dead reckoning (location sensing), as in one of the experiments from the original paper. In pedestrian dead reckoning, a model utilizes the inertial measurement unit (IMU) signals to estimate the 2D walking trajectory of phone users. This task is very challenging due to the localization error is cumulative.

In the experiment, we use TCN [1] as baseline model and show TASFAR can further improves its performance on each user.

This demo consists of two parts:
- Pseudo-label generation of TASFAR
- Training on pseudo-labels
- Testing on adapted models

## Kick start
### Environment setup
```
conda create -n tasfar_demo_env python=3.11  # If you are using anaconda 
conda activate tasfar_demo_env  # Activate the environment
```
Dependencies (more details can be found in ./requirements.txt)
```
matplotlib==3.5.2
numpy==1.21.6
ortools==9.0.9048
pandas==1.3.5
Pillow==9.0.1
pyparsing==3.0.9
torch==1.11.0
torchaudio==0.11.0
torchvision==0.12.0
```
### Pseudo-label testing
```
cd ./source/
# Generating pseudo label for user1, we show [user1, user2, user3] considering github storage
# -d refers to computing device
python ./gen_pseudo_label.py -u user1 -d cpu  
```
Sample result for user 1
```
----------------------------------------------------------------
Information of user1:
Trajectory length: 486.07m
Time period: 456s
Number of steps (2s): 228
Uncertain data ratio: 12.28%
Average step error (STE) before adaptation: 0.620m
Average step error (STE) after adaptation: 0.553m
STE reduction rate: 10.89%
----------------------------------------------------------------
```

### Testing 
In this part, we demo test results using provided adapted model (from ./model/user1_model.pt).
```
cd ./source/
# Generating pseudo label for user1, 
python ./test.py -u user1 -d cpu  
```
Results: although both outputs (before/after adaptation) deviate from the ground truth due to error accumulation over long distances (~400m), TASFAR optimizes the trajectory shapes through its adaptation approach.
- User 1

<img src="https://github.com/Siriusize/TASFAR_demo/blob/main/figure/user1.png" alt="user1" width="400"/>

- User 2

<img src="https://github.com/Siriusize/TASFAR_demo/blob/main/figure/user2.png" alt="user2" width="400"/>

- User 3

<img src="https://github.com/Siriusize/TASFAR_demo/blob/main/figure/user3.png" alt="user3" width="400"/>

### Training 
We also provide demo for training. 
```
cd ./source/
# Generating pseudo label for user1, 
python ./train.py -u user1 -d cpu 
```
The module will automatically test on the trained model with statistics and figures shown as above.
```
------------------------------------------------------------------
Relative trajectory error (RTE) of adaptation set (origin): 2.460
Relative trajectory error (RTE) of adaptation set (TASFAR): 2.080
Relative trajectory error (RTE) of test set (origin): 2.387
Relative trajectory error (RTE) of test set (TASFAR): 1.956
Trajectory visualization has been saved to '../figure/user1'
------------------------------------------------------------------
```


## Reference
[1] Herath S, Yan H, Furukawa Y. Ronin: Robust neural inertial navigation in the wild: Benchmark, evaluations, & new methods[C]//2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020: 3146-3152.