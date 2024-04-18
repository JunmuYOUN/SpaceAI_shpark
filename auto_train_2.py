import subprocess
# wave_set = ['094','131','133','171','177','180','195','202','211','256','284','304','335','368']
wave_set = ['094','133']
lr_1 = [0.0001]
lr_2 = [0.00003]
# activation = ["ReLU"]
CC_loss = [0.1,0.3,0.5, 0.7, 1]

for i in wave_set:
    if i in ["094", "133"]: 
        for j in lr_1:
            for k in CC_loss:
                print("Train wavelength", i)
                print("Learning rate", j)
                print("CC_loss", k)
                subprocess.call('python FCN_torch_Train.py -w {} -lr {} -a {}'.format(i, j, k), shell=True)
    else:
        for j in lr_2:
            for k in activation:
                print("Train wavelength", i)
                print("Learning rate", j)
                print("CC_loss", k)
                subprocess.call('python FCN_torch_Train.py -w {} -lr {} -a {}'.format(i, j, k), shell=True)