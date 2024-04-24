
# What is the TissueSearchAgent
The TissueSearchAgent is a Deep Reinforcement Learning model aimed at efficiently removing the white space patches from a WSI. 
This agent is targeted towards CPath developers to expedite the pre-processing required for WSI-based projects. 

# Requirements
- Python version 3.10
### Libraries:
- TIAToolbox
- Openslide
- OpenCV 
- PIL
- Numpy
- PyTorch
- Torchvision


# How to Train
To train the TissueSearchAgent, "TissueSearchAgent-Final.py" should be ran. 

This file takes as input:
1. Path to WSIs
2. Second Path to WSIs (if using a predefined train/test split)
3. Path to ouput the trained models
4. Path to a workspace (used for creationg of WSI patches)

# Dataset
https://camelyon16.grand-challenge.org/Data/
