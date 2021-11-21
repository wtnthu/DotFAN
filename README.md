# [DotFAN: A Domain-transferred Face Augmentation Network for Pose and Illumination Invariant Face Recognition](https://www.ee.nthu.edu.tw/cwlin/pub/tip_face_2021.pdf)
The source code for the paper: H.-C. Shao, K.-Y. Liu, W.-T. Su, C.-W. Lin, and J. Lu “DotFAN: A domain-transferred face augmentation net" IEEE Trans. Image Processing, vol. 30, pp. 8759−8772, Nov. 2021 [Paper link](https://www.ee.nthu.edu.tw/cwlin/pub/tip_face_2021.pdf).

## Quick Start
### Installation
**1.Install dependency**
pip install -r requirement.txt
    
**2.Download the face recognition and 3D face model (3DMM) model to ./data/pretrained_ckpt as below:**
https://drive.google.com/drive/folders/1DuY5rGwCeZtrNfYg6tLkLc2T1O0rGu1M?usp=sharing

**3.Download the pretrained model to ./v20/models as below:**
https://drive.google.com/drive/folders/11F33tinezQA8TrqHzrH3jZMgGPj11Tye?usp=sharing

### Run the code
All hyper-parameters can be modified in config_pose_TW1.yml

**1.Run shell/train.sh to train the DotFAN model. when training v20, batchsize is 16.**

**2.Run shell/val.sh to synthesize the multiple output in a image.**

**3.Run shell/norm.sh to frontalize the profile face images.**

**4.Run shell/augmentation.sh to synthesize face images according to target pose images in a folder such as data/ref_pie_pose.**
