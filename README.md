# DotFAN
The source code for the paper "DotFAN: A Domain-transferred Face Augmentation Network for Pose and Illumination Invariant Face Recognition"

Quick Start
Installation

Prepare the feature extracted from any face recognition model 
https://drive.google.com/drive/folders/1DuY5rGwCeZtrNfYg6tLkLc2T1O0rGu1M?usp=sharing




All hyper-parameters can be modified in config_pose_TW1.yml
1.Execute requirements.txt to install lib
2.Run shell/train.sh to train the DotFAN model. when training v20, batchsize is 16.
3.Run shell/val.sh to synthesize the multiple output in a image.
4.Run shell/norm.sh to frontalize the profile face images.
5.Run shell/augmentation.sh to synthesize face images according to target pose images in a folder such as data/ref_pie_pose.
