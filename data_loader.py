import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from skimage import io
import numpy as np
from pathlib import Path
def read_folder(Folder_root,WithFolder_root=False):
    file_list = []
    for dirname, dirnames, filenames in os.walk(Folder_root):
        for filename in filenames:
            if WithFolder_root:
                file_list.append(os.path.join(dirname, filename))
            else:
                file_list.append(os.path.join(dirname, filename).split(Folder_root+"/")[1])
    return file_list
class pie_dataset(Dataset):
    def __init__(self,config, image_path, transform, mode,gray):
        self.mask_transform = transforms.Compose([
            transforms.Resize([112, 112],Image.BICUBIC),
            transforms.ToTensor(),
            ])
        self.config = config
        self.image_path = image_path
        if gray:
            self.input_channel = 1
        else:
            self.input_channel = 3
        self.transform = transform
        self.mode = mode
        #self.lines = open("data/"+mode+".txt", 'r').readlines()
        self.lines = Path("data/"+mode+".txt").read_text().strip().split('\n')
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')


        self.num_data = len(self.filenames_path)
        print("len of data:" + str(self.num_data))

    def preprocess(self):
        self.filenames_path = []
        self.filenames_id=[]
        self.filenames_label = []


        lines = self.lines[1:]
        if 'train' in self.mode:
            random.shuffle(lines)   # random shuffling
        #lines.reverse()
        for i, line in enumerate(lines):

            splits = line.split()
            if 'test' in self.mode:
                filename = line
            else:
                filename = splits[0]
            label = []
            id=0
            if self.mode == 'train_illumination':
                values = splits[1:]
                label.append(0)
                for idx, value in enumerate(values):
                        if value == '1':
                            label.append(1)
                        else:
                            label.append(0)

            elif self.mode == 'train':
                label.append(1)
                for i in range(13):
                    label.append(0)

            self.filenames_id.append(id)
            self.filenames_path.append(filename)
            self.filenames_label.append(label)
        if not self.mode == 'train':
            imgs_list = read_folder(self.config.taget_pose,WithFolder_root=True)

            list.sort(imgs_list)
            print(imgs_list)
            if self.input_channel ==1:
                self.pie_pose = [np.uint8(Image.open(path).convert("L"))[...,np.newaxis].transpose( (2 ,0 , 1)).astype('float32') / 255
                                 for path in imgs_list]
                self.pie_pose = np.array(self.pie_pose)
                self.pie_pose = (self.pie_pose-0.5)*2
            else:
                self.pie_pose = [np.uint8(Image.open(path)).transpose((2, 0, 1)).astype('float32') / 255
                                 for path in imgs_list]#4.5 2.7
                self.pie_pose = np.array(self.pie_pose)
                self.pie_pose = (self.pie_pose - 0.5) * 2
    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'train_illumination':
            image = Image.open(os.path.join(self.image_path, self.filenames_path[index]))
            use_mask = False # If True prepare the mask of each images by 3DMM model in
            if use_mask:
                mask = io.imread(os.path.join(self.image_path, 'mask_precise',self.filenames_path[index]))
                mask = (mask>=10).astype("float32")*255
                mask = mask.convert("L")
                mask = Image.fromarray(np.uint8(mask))
            else:
                mask = Image.fromarray(np.zeros_like(np.array(image)))
            if self.input_channel==1:
                image = image.convert("L")

            label = self.filenames_label[index]
            return self.transform(image),torch.FloatTensor(label),self.mask_transform(mask)
        elif  self.mode == 'val':
            image = Image.open(os.path.join(self.image_path, self.filenames_path[index]))
            if self.input_channel == 1:
                image = image.convert("L")

            return self.transform(image), self.filenames_path[index], torch.tensor(self.pie_pose)
        else:

            image = Image.open(os.path.join(self.image_path, self.filenames_path[index]))

            if self.input_channel==1:
                image = image.convert("L")

            return self.transform(image), self.filenames_path[index],torch.tensor(self.pie_pose)

    def __len__(self):
        return self.num_data



def get_loader(config,image_path, crop_size, image_size, batch_size, dataset, mode='train'):
    """Build and return data loader."""
    osize = [image_size, image_size]
    if mode == 'train' or mode =='trainB' :
        transform = transforms.Compose([
            transforms.Resize(osize, Image.BICUBIC),
            transforms.RandomCrop(crop_size),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.Resize([crop_size, crop_size],Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    dataset = pie_dataset(config,image_path, transform, mode,config.gray)
    shuffle = False
    drop_last = False
    if 'train' in mode or 'test' in mode or 'val' in mode:
        shuffle = True
    if 'train' in mode:
        drop_last = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=shuffle,
                             pin_memory=True,
                             drop_last=drop_last
                             )

    return data_loader
