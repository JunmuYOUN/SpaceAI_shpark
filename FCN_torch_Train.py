import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset, DataLoader

# import torchvision
# import torchvision.transforms as transform
# from torchvision.transforms import ToTensor

from tqdm import tqdm
from glob import glob
from torchsummary import summary

import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

from pytorchtools import EarlyStopping



parser = argparse.ArgumentParser(description='')
parser.add_argument('-w', '--wave')
parser.add_argument('-act', '--Activationft',default="ReLU",  type=str)
parser.add_argument('-lr', '--learningrate', default=0.01 , type=float)
parser.add_argument('-a', '--alpha', default=0, type=float)
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

current_path = os.getcwd()
early_stopping = EarlyStopping(patience = 30, verbose = True,  path = current_path+'/{}/checkpoint/FCN_best/'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha)))







class CC_loss(nn.Module):
    def __init__(self):
        super(CC_loss, self).__init__()

    def forward(self, fake, target):
        rd = target - torch.mean(target)
        fd = fake - torch.mean(fake)

        r_num = torch.sum(rd * fd)
        r_den = torch.sqrt(torch.sum(rd ** 2)) * torch.sqrt(torch.sum(fd ** 2))
        pcc_val = r_num / (r_den + 1e-10)

        loss_cc = 1.0 - pcc_val

        return loss_cc




# Dataset
class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        
        self.input_format = 'npy'
        self.target_format = 'npy'
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        self.label_path_list = sorted(glob(os.path.join(self.input_dir,'*.npy')))
        self.target_path_list = sorted(glob(os.path.join(self.target_dir,'*.npy')))
        print(len(self.label_path_list), len(self.target_path_list))
        
        
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
        
        #Train target
        IMG_target = np.load(self.target_path_list[idx], allow_pickle=True)
        target_array = IMG_target
        target_shape = target_array.shape
        target_tensor = torch.tensor(target_array, dtype=torch.float32)

        return label_tensor, target_tensor
    
    
    
    
    
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


train_dataset = CustomDataset("C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/x_train".format(args.wave),
                              "C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/y_train".format(args.wave))

validation_dataset = CustomDataset("C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/x_valid".format(args.wave),
                                   "C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/y_valid".format(args.wave))


# label_path_list = sorted(glob(os.path.join("C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/x_train".format(args.wave),'*.npy')))
# target_path_list = sorted(glob(os.path.join("C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/y_train",'*.npy')))
# if (len(label_path_list) != len(target_path_list)):
#     print(len(label_path_list), len(target_path_list))
#     raise Exception("Dataset might not be correspond")

batch_size = 32  # You can adjust the batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)


model = FCN().to(device)
opt = torch.optim.Adam(model.parameters(), lr= args.learningrate, betas=(0.9,0.999))
loss_fn_MSE = nn.MSELoss()
loss_fn_CC = CC_loss()


summary(model, (360,1), device=device.type)
# print(model)


num_epochs = 300


# TensorBoard 설정
log_dir = current_path+"/{}/logs/".format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha))  # 로그를 저장할 디렉토리 경로
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    validation_loss = 0.0
    total_loss = 0.0
    k=0

    for batch_input_1, batch_target in tqdm(train_loader):
            batch_input_1, batch_target = batch_input_1.to(device), batch_target.to(device)
            predictions = model(batch_input_1)
            MSE_loss = loss_fn_MSE(predictions, batch_target)
            CC_loss = loss_fn_CC(predictions, batch_target)
            # print(type(args.alpha))
            
            combined_loss = (1 - args.alpha) * MSE_loss + (args.alpha) * CC_loss
            
            opt.zero_grad()
            combined_loss.backward()
            opt.step()

            total_loss += combined_loss.item()


    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    
    if epoch % 3 == 0: 
        try:
            torch.save(model.state_dict(),
                       current_path+'/{}/checkpoint/FCN/model_checkpoint_{:0>3}.pth'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha), str(epoch)))
        except:
            os.makedirs(current_path+'/{}/checkpoint/FCN/'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha)))
            torch.save(model.state_dict(),
                       current_path+'/{}/checkpoint/FCN/model_checkpoint_{:0>3}.pth'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha), str(epoch)))

    

    # Validation
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch_input, batch_target in validation_loader:
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            predictions = model(batch_input)
            val_MSE_loss = loss_fn_MSE(predictions, batch_target)
            val_CC_loss = loss_fn_CC(predictions, batch_target)
            
            val_combined_loss = (1 - args.alpha) * MSE_loss + (args.alpha) * CC_loss
            
            validation_loss += val_combined_loss.item()

    # Calculate
    avg_validation_loss = validation_loss / len(validation_loader)
    print(f"Validation Loss: {avg_validation_loss:.4f}")
    writer.add_scalar('Valid/Loss', validation_loss, epoch)
    
    ### early stopping 여부를 체크하는 부분 ###
    try:
        early_stopping(avg_validation_loss, model,str(epoch) ) # 현재 과적합 상황 추적
        
    except:
        os.makedirs(current_path+'/{}/checkpoint/FCN_best/'.format(args.wave+'_'+str(args.learningrate)+'_'+args.Activationft+'_'+str(args.alpha)))
        early_stopping(avg_validation_loss, model,str(epoch)) # 현재 과적합 상황 추적
        
    if early_stopping.early_stop: # 조건 만족 시 조기 종료
        break
    