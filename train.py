import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime
from src import UNet,deeplabv3_resnet50
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
from tqdm import tqdm
from PIL import Image
import transforms as T
import torch.nn.functional as F
import shutil


class MultiFolderImageDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.folders = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
        self.transform = transform
        self.current_folder = None
        self.image_list = None

    def set_folder(self, folder):
        self.current_folder = folder
        self.image_list = sorted(os.listdir(folder))

    def __len__(self):
        if self.image_list is None:
            return 0
        return len(self.image_list) - 1

    def __getitem__(self, idx):
        img1_path = os.path.join(self.current_folder, self.image_list[idx])
        img2_path = os.path.join(self.current_folder, self.image_list[idx + 1])
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        if img1 is None or img2 is None:
            raise ValueError(f"Failed to load images at {img1_path} or {img2_path}")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, self.image_list[idx + 1]

#def create_result_folder(base_dir, episode, folder_name):
    #episode_folder = os.path.join(base_dir, f'{folder_name}_episode_{episode}')
    #anomalies_folder = os.path.join(episode_folder, 'anomalies')
    #optimized_folder = os.path.join(episode_folder, 'optimized')
    
    #os.makedirs(anomalies_folder, exist_ok=True)
    #os.makedirs(optimized_folder, exist_ok=True)
    
    #return anomalies_folder, optimized_folder

def detect_anomalies_statistical(images, threshold):
    anomalies = []
    anomaly_masks = []
    for i in range(1, len(images)):
        prev_img = images[i-1]
        next_img = images[i]
        
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        
        anomaly_mask = magnitude > (mean_mag + threshold * std_mag)
        
        kernel = np.ones((5, 5), np.uint8)
        anomaly_mask = cv2.dilate(anomaly_mask.astype(np.uint8), kernel, iterations=2)
        
        anomaly_masks.append(anomaly_mask)
        
        anomaly_img = cv2.cvtColor((next_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        anomaly_img[anomaly_mask.astype(bool)] = [0, 0, 255]
        
        anomalies.append(anomaly_img)
    return anomalies, anomaly_masks

def dice_similarity_coefficient(img1, img2):
    intersection = (img1 * img2).sum()
    if img1.sum() + img2.sum() == 0:
        return torch.tensor(1.0)
    return (2. * intersection) / (img1.sum() + img2.sum())

def binarize_image(img, threshold=0.5):
    return (img > threshold).float()

class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * 512 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def calculate_initial_avg_abs_dsc_diff(dataloader, device):
    prev_dscs = []
    diffs = []
    prev_dsc = None
    for i, (img1, img2, img2_name) in enumerate(dataloader):
        img1, img2 = img1.to(device), img2.to(device)
        bin_img1 = binarize_image(img1)
        bin_img2 = binarize_image(img2)
        dsc = dice_similarity_coefficient(bin_img1, bin_img2)
        if prev_dsc is not None:
            abs_dsc_diff = torch.abs(dsc - prev_dsc)
            diffs.append(abs_dsc_diff.item())
        prev_dsc = dsc
        prev_dscs.append(dsc)
    if len(diffs) == 0:
        return float('nan'), prev_dscs
    return np.mean(diffs), prev_dscs



class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 608
    crop_size = 576

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def train_unet(path_unet_weight, max_epoch, model, optimizer_unet, unet_train_loader, device,  lr_scheduler):
    for epoch in range(max_epoch):
        mean_loss, lr = train_one_epoch(model, optimizer_unet, unet_train_loader, device, epoch, num_classes=2,
                                        lr_scheduler=lr_scheduler, print_freq=10, scaler=None)
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer_unet.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch
                     }

    torch.save(save_file, os.path.join(unet_weights_path,"drl_unet"+str(max_epoch)+".pth"))

def test_unet(unet_weights_path, unet_img_path, unet_result_path, max_epoch):
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    model.load_state_dict(torch.load(os.path.join(unet_weights_path,"drl_unet"+str(max_epoch)+".pth"), map_location='cpu')['model'])
    model.to(device)
    pathDir = os.listdir(unet_img_path)    #取图片的原始路径
    for name in tqdm(pathDir):
     # load image
         img=os.path.join(unet_img_path,name)      
         original_img = Image.open(img).convert('RGB')
       
     # from pil image to tensor and normalize
         data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
         img = data_transform(original_img)
     # expand batch dimension
         img = torch.unsqueeze(img, dim=0)
         model.eval()  # 进入验证模式   
         with torch.no_grad():
        # init model
             img_height, img_width = img.shape[-2:]
             init_img = torch.zeros((1, 3, img_height, img_width), device=device)
             model(init_img)

             output = model(img.to(device))              
             #print(output['out'].argmax(1).squeeze(0))
             output = output['out']
             #print(output)
             output_softmax = F.softmax(output, dim=1) #应用softmax函数

             pixel_probabilities = output_softmax[0, 0, :, :].cpu().detach().numpy()
             pixel_probabilities = 1-pixel_probabilities
             #print(pixel_probabilities)
             cv2.imwrite(os.path.join(unet_result_path,name),(pixel_probabilities*255).astype(np.uint8))


def train_drl(epoch,agent, dataloader, optimizer_drl, criterion, num_episodes, threshold, device, result_base_dir, model_save_dir):
    best_reward = -float('inf')
    closest_diff = float('inf')
    closest_diff_model_state = None

    for episode in range(num_episodes):
        total_loss = 0.0
        total_reward = 0.0

        folder = random.choice(dataloader.dataset.folders)
        dataloader.dataset.set_folder(folder)
        folder_name = os.path.basename(folder)
        
        #anomalies_folder, optimized_folder = create_result_folder(result_base_dir, episode, folder_name)
        
        initial_avg_abs_dsc_diff, prev_dscs = calculate_initial_avg_abs_dsc_diff(dataloader, device)
        if np.isnan(initial_avg_abs_dsc_diff):
            print(f"No valid DSC differences found in folder: {folder_name}")
            continue
        else:
            print(f"Initial average absolute DSC difference: {initial_avg_abs_dsc_diff:.4f}")

        new_prev_dscs = []
        for i, (img1, img2, img2_name) in enumerate(dataloader):
            optimizer_drl.zero_grad()
            
            img1, img2 = img1.to(device), img2.to(device)
            
            anomalies, anomaly_masks = detect_anomalies_statistical([img1.cpu().numpy().squeeze(), img2.cpu().numpy().squeeze()], threshold)
            anomaly_mask = torch.tensor(anomaly_masks[0], dtype=torch.float).to(device)
            
            bin_img1 = binarize_image(img1)
            bin_img2 = binarize_image(img2)

            state = torch.cat((img1, img2), dim=0).unsqueeze(0).float().to(device)
            action_values = agent(state)
            action = torch.argmax(action_values).item()
            
            random_change = torch.rand_like(anomaly_mask) * 0.3
            if action == 0:
                img2 = torch.clamp(img2 + random_change * anomaly_mask, 0, 1)
            else:
                img2 = torch.clamp(img2 - random_change * anomaly_mask, 0, 1)
            
            new_bin_img2 = binarize_image(img2)
            new_dsc = dice_similarity_coefficient(bin_img1, new_bin_img2)
            new_prev_dscs.append(new_dsc)
            
            if len(new_prev_dscs) > 1:
                new_abs_dsc_diff = torch.abs(new_prev_dscs[-1] - new_prev_dscs[-2])
            else:
                new_abs_dsc_diff = torch.tensor(0.0)
            
            reward = 1 if 0 < new_abs_dsc_diff <= initial_avg_abs_dsc_diff else -1
            total_reward += reward
            
            loss = criterion(action_values, torch.tensor([reward], dtype=torch.float).to(device))
            loss.backward()
            optimizer_drl.step()
            
            total_loss += loss.item()

            episode_folder = os.path.join(result_base_dir, f'{folder_name}')
            #print(episode_folder)
            os.makedirs(episode_folder, exist_ok=True)
            #anomaly_img = anomalies[0]
            #cv2.imwrite(os.path.join(anomalies_folder, f'anomaly_{i}.png'), anomaly_img)
            optimized_img = (new_bin_img2.cpu().numpy().squeeze() * 255).astype(np.uint8)
            #print(img2_name[0])
            cv2.imwrite(os.path.join(episode_folder, img2_name[0]), optimized_img)
            
        print(f"Episode {episode + 1}/{num_episodes}, Loss: {total_loss:.4f}, Total Reward: {total_reward:.4f}")
        
        # 保存最优模型
        if total_reward > best_reward:
            best_reward = total_reward
            model_save_path = os.path.join(model_save_dir, f'best_model_epoch_{epoch}.pth')
            torch.save(agent.state_dict(), model_save_path)
            print(f"New best model saved at episode {episode + 1} with reward: {total_reward:.4f}")

        # 保存 new_abs_dsc_diff 与 initial_avg_abs_dsc_diff 最接近的模型
        diff = torch.abs(new_abs_dsc_diff - initial_avg_abs_dsc_diff).item()
        if diff < closest_diff:
            closest_diff = diff
            closest_model_path = os.path.join(model_save_dir, f'closest_model_epoch_{epoch}.pth')
            torch.save(agent.state_dict(), closest_model_path)
            print(f"New closest model saved at episode {episode + 1} with diff: {diff:.4f}")
    return episode_folder


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_max_epoch=100
    DRL_max_epoch=10
    mil_folder = "./logist"
    mask_folder = "./mask"
    #drl_result_dir = "./result-unet-drl"
    #drl_model_save_dir = "./model-unet-drl"   
    save_dir='results_0.6'
    save_results_dir=os.path.join(save_dir,'results')
    save_model_dir=os.path.join(save_dir,'weights')
    
    
    os.makedirs(save_results_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    agent = DQNAgent().to(device)
    optimizer_drl = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()


    model = UNet(in_channels=3, num_classes=2, base_c=32)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer_unet = torch.optim.SGD(
         params_to_optimize,
         lr=0.001, momentum=0.9, weight_decay=1e-4
    )

    unet_weights_path = './unet_weights'
   
    total_epoch=100 #整体训练次数

    for epoch in range(total_epoch):

    #训练drl
        if epoch==0:
            dataset = MultiFolderImageDataset(mil_folder, transform)
        else:
            #print(unet_result_path.split('/')[:-1])
            dataset = MultiFolderImageDataset(os.path.join(save_results_dir,str(epoch-1),'UNET'), transform)

        drl_result_dir = os.path.join(save_results_dir,str(epoch),'DRL')
        os.makedirs(drl_result_dir, exist_ok=True)



        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        episode_folder=train_drl(epoch,agent, dataloader, optimizer_drl, criterion, num_episodes=DRL_max_epoch, threshold=2.0, device=device, result_base_dir=drl_result_dir, model_save_dir=save_model_dir)
    
    #训练unet
    # using compute_mean_std.py
        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        
        files_to_copy=set(os.listdir(os.path.join(mil_folder, os.path.basename(episode_folder))))-set(os.listdir(episode_folder))
        #print(files_to_copy)
        for file_name in files_to_copy:
            temp_img=cv2.imread(os.path.join(mil_folder, os.path.basename(episode_folder),file_name))
            temp_img[temp_img>127]=255
            temp_img[temp_img<=127]=0
            cv2.imwrite(os.path.join(episode_folder, file_name),temp_img)

            #shutil.copy(os.path.join(), file_name), )
        
        if random.random()<=0.6:
            unet_train_folder=episode_folder
        else:
            unet_train_folder=os.path.join(mask_folder,os.path.basename(episode_folder))
        print(unet_train_folder)
        unet_train_dataset = DriveDataset(unet_train_folder,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

        unet_train_loader = torch.utils.data.DataLoader(unet_train_dataset,
                                               batch_size=8,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=False,
                                               collate_fn=unet_train_dataset.collate_fn)

        lr_scheduler = create_lr_scheduler(optimizer_unet, len(unet_train_loader),unet_max_epoch, warmup=True)   

        unet_weights_path = os.path.join(save_model_dir,str(epoch))
        os.makedirs(unet_weights_path, exist_ok=True)

        train_unet(unet_weights_path, unet_max_epoch, model, optimizer_unet, unet_train_loader, device, lr_scheduler)


        for test_folder in os.listdir('./image'):
            unet_img_path = os.path.join('./image',test_folder)
            unet_result_path = os.path.join(save_results_dir,str(epoch),'UNET',test_folder)
            os.makedirs(unet_result_path, exist_ok=True)

            test_unet(unet_weights_path, unet_img_path, unet_result_path, unet_max_epoch)
        
