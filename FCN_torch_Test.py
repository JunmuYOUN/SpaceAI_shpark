import torch
import torch.nn as nn
import os
from os.path import split, splitext

import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transform
from torchvision.transforms import ToTensor
from tqdm import tqdm
from glob import glob
from torchsummary import summary

import numpy as np


import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='')
parser.add_argument('-w', '--wave')
parser.add_argument('-act', '--Activationft', default="ReLU" , type=str)
parser.add_argument('-lr', '--learningrate', default=0.01 , type=float)
parser.add_argument('-a', '--alpha', default=0, type=float)
args = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

# Dataset
class CustomDataset(Dataset):
    def __init__(self, input_dir):
        
        self.input_format = 'npy'
        self.input_dir = input_dir
        
        self.label_path_list = sorted(glob(os.path.join(self.input_dir,'*.npy')))
        print(len(self.label_path_list))
        
    def __len__(self):
        return len(self.label_path_list)

    def __getitem__(self, idx):
        
        list_transforms = []  #?
        list_transforms += [] #?
        
        #Train input
        IMG_label = np.load(self.label_path_list[idx], allow_pickle=True)
        label_array = IMG_label
        label_shape = label_array.shape
        label_tensor = torch.tensor(label_array, dtype=torch.float32)
        
        
        
        # #Train target
        # IMG_target = np.load(self.target_path_list[idx], allow_pickle=True)
        # target_array = IMG_target
        # target_shape = target_array.shape
        # target_tensor = torch.tensor(target_array, dtype=torch.float32)

        return label_tensor, splitext(split(self.label_path_list[idx])[-1])[0]
    
    
#model
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.init_block = 360
        
        self.flatten = nn.Flatten()
        
        linear1 = nn.Linear(in_features = self.init_block, out_features = self.init_block*8)
        linear2 = nn.Linear(in_features = self.init_block*8, out_features = self.init_block*16)
        linear3 = nn.Linear(in_features = self.init_block*16, out_features = self.init_block*16)
        linear4 = nn.Linear(in_features = self.init_block*16, out_features = self.init_block*8)
        linear5 = nn.Linear(in_features = self.init_block*8, out_features = self.init_block)
        
        bn1 = nn.BatchNorm1d(self.init_block*8)
        bn2 = nn.BatchNorm1d(self.init_block*16)
        bn3 = nn.BatchNorm1d(self.init_block*16)
        bn4 = nn.BatchNorm1d(self.init_block*8)
        
        self.linear_stack = nn.Sequential(linear1, bn1, self.activation(args.Activationft),
                                               linear2, bn2, self.activation(args.Activationft),
                                               linear3, bn3, self.activation(args.Activationft),
                                               linear4, bn4, self.activation(args.Activationft),
                                               linear5)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
    
    
    def activation(self, Activationft):
        if Activationft == "SiLU":
            return nn.SiLU()
        elif Activationft == "SELU":
            return nn.SELU()
        elif Activationft == "ReLU":
            return nn.ReLU()

    
    
test_dataset = CustomDataset("C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/x_test".format(args.wave))
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)    

#batch_size = 1  # You can adjust the batch size as needed


###
dir_model = './{}/checkpoint/FCN/*.pth'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha))
# ITERATIONs = sorted([os.path.basename(x).split('_')[0] for x in glob(dir_model)])
ITERATIONs = sorted([os.path.basename(x).split('.')[-2] for x in glob(dir_model)])             

# print(ITERATIONs)

# NUMBER = sorted([x.split('_')[-1] for x in ITERATIONs])             

# print(NUMBER)


# IF Best
ITERATIONs = ["model_checkpoint_Best"]


for ITERATION in ITERATIONs:

    path_model = './{}/checkpoint/FCN_best/{}'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha),str(ITERATION))
    dir_image_save = './{}/checkpoint/FCN_result/image_{}'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha),str(ITERATION))
    model = FCN().to(device)
    model.load_state_dict(torch.load('./{}/checkpoint/FCN_best/{}.pth'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha),str(ITERATION))))
    
    if os.path.isdir(dir_image_save) == True:
        pass
    else:                                             
        os.makedirs(dir_image_save, exist_ok=True)
        
        with torch.no_grad():
                        model.eval()
                        for input, name in tqdm(test_data_loader):
                            
                            input = input.to(device)
                            fake = model(input)
                            np_fake = fake.cpu().numpy().squeeze()
                            np.save(os.path.join(dir_image_save, str(name[0]) + '_AI.npy'), np_fake, allow_pickle=True)    
    