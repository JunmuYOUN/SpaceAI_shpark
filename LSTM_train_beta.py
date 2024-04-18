import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset, DataLoader

# import torchvision
# import torchvision.transforms as transform
# from torchvision.transforms import ToTensor

from tqdm import tqdm
from glob import glob
from torchinfo import summary

import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

from pytorchtools import EarlyStopping

# Argument parsing setup
parser = argparse.ArgumentParser(description='')
parser.add_argument('-w', '--wave')
parser.add_argument('-act', '--Activationft', default="ReLU", type=str)
parser.add_argument('-lr', '--learningrate', default=0.01, type=float)
parser.add_argument('-a', '--alpha', default=0, type=float)
args = parser.parse_args()

# Environment setup for CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

# Early stopping setup
current_path = os.getcwd()
early_stopping = EarlyStopping(patience=30, verbose=True, path=current_path + '/{}/checkpoint/FCN_best/'.format(args.wave + '_' + str(args.learningrate) + '_' + args.Activationft + '_' + str(args.alpha)))

# Loss definition
class CC_loss(nn.Module):
    def forward(self, fake, target):
        rd = target - torch.mean(target)
        fd = fake - torch.mean(fake)
        r_num = torch.sum(rd * fd)
        r_den = torch.sqrt(torch.sum(rd ** 2)) * torch.sqrt(torch.sum(fd ** 2))
        pcc_val = r_num / (r_den + 1e-10)
        loss_cc = 1.0 - pcc_val
        return loss_cc

# Dataset definition
class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.label_path_list = sorted(glob(os.path.join(self.input_dir, '*.npy')))
        self.target_path_list = sorted(glob(os.path.join(self.target_dir, '*.npy')))

    def __len__(self):
        return len(self.label_path_list)

    def __getitem__(self, idx):
        label_array = np.load(self.label_path_list[idx], allow_pickle=True)
        target_array = np.load(self.target_path_list[idx], allow_pickle=True)
        return torch.tensor(label_array, dtype=torch.float32), torch.tensor(target_array, dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        # self.flatten = nn.Flatten()
        self.lstm_size = 256  # Number of features in the hidden state
        self.num_layers = 4   # Number of recurrent layers
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=self.lstm_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(self.lstm_size * 2)
        self.lstm2 = nn.LSTM(input_size=self.lstm_size*2, hidden_size=self.lstm_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(self.lstm_size * 2)
        self.lstm3 = nn.LSTM(input_size=self.lstm_size*2, hidden_size=self.lstm_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.ln3 = nn.LayerNorm(self.lstm_size * 2)
        self.fc = nn.Linear(self.lstm_size*2, 360)  # Output layer
        self.bn1 = nn.BatchNorm1d(360)

    def forward(self, x):

        x = x.unsqueeze(-1)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.lstm_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.lstm_size).to(device)
        out, (hn, cn) = self.lstm1(x, (h0, c0))
        out = self.ln1(out)
        out, (hn, cn) = self.lstm2(out, (hn, cn))
        out = self.ln2(out)
        out, (hn, cn) = self.lstm3(out, (hn, cn))
        out = self.ln3(out)
        out = self.fc(out[:, -1, :])  # we only want the last time step output
        out = self.bn1(out)
        return out

# Dataset and DataLoader instantiation
train_dataset = CustomDataset("C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/x_train".format(args.wave),
                              "C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/y_train".format(args.wave))

validation_dataset = CustomDataset("C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/x_valid".format(args.wave),
                                   "C:/Users/user/Desktop/Space_AI_shpark/dataset/Numpy_Resample_Norm/dataset_{}/y_valid".format(args.wave))


batch_size = 32  # You can adjust the batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)


# Model, optimizer, and loss function instantiation
model = LSTMModel().to(device)
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
    