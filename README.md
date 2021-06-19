# DotFAN
The source code for the paper "DotFAN: A Domain-transferred Face Augmentation Network for Pose and Illumination Invariant Face Recognition"

## Quick Start
### Installation
1.Install dependency
pip install -r requirement.txt

2.Download the face recognition and 3D face model (3DMM) model to ./data/pretrained_ckpt as below:
https://drive.google.com/drive/folders/1DuY5rGwCeZtrNfYg6tLkLc2T1O0rGu1M?usp=sharing

### Run the code
All hyper-parameters can be modified in config_pose_TW1.yml
Run shell/train.sh to train the DotFAN model. when training v20, batchsize is 16.
Run shell/val.sh to synthesize the multiple output in a image.
Run shell/norm.sh to frontalize the profile face images.
Run shell/augmentation.sh to synthesize face images according to target pose images in a folder such as data/ref_pie_pose.
