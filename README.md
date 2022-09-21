# On-ramp Merging
The code need install sumo https://sumo.dlr.de/docs/Downloads.php

# Dependencies
numpy==1.17.0
torch==0.4.1.post2
matplotlib==3.0.0
PyVirtualDisplay==0.2.1
gym==0.10.9
nn_builder==1.0.5
tensorflow==2.0.0a0
tensorboardx==2.1	
python==3.7.10	
pytorch==1.7.1	
pybullet==3.1.0	
SUMO==0.19.0

# Training
results/ramp_mergingRun.py

The tensorboard file and model file will be saved in ./results/log and ./results/Models, 

# Testing
results/ramp_merge_text.py

The policy our trained will be in ./results/Models.


# Thanks
Thanks for code:
https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
