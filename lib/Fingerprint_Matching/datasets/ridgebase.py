from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class RidgeBase(Dataset):
    def __init__(self, split = "train"):
        fingerdict = {
            "Index": 0,
            "Middle":1,
            "Ring": 2,
            "Little": 3
        }
        
        self.split = split
        
        self.transforms ={
            "train": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    transforms.RandomResizedCrop(224),
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )]),
            "test": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224,224)),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                    ])
        }
        
        self.train_path = {
            "contactless": "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/Task1/Train/Contactless",
            "contactbased": "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/Task1/Train/Contactbased"
        }
        self.test_path = {
            "contactless": "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/Task1/Test/Contactless",
            "contactbased": "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/Task1/Test/Contactbased"
        }
        self.train_files = {
            "contactless": [self.train_path["contactless"] + "/" + f for f in os.listdir(self.train_path["contactless"]) if f.endswith('.png')],            
            "contactbased": [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.train_path["contactbased"]) for f in filenames if os.path.splitext(f)[1] == '.bmp']        
        }
        self.test_files = {
            "contactless": [self.test_path["contactless"] + "/" + f for f in os.listdir(self.test_path["contactless"]) if f.endswith('.png')],            
            "contactbased": [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.test_path["contactbased"]) for f in filenames if os.path.splitext(f)[1] == '.bmp']     
        }
        
        self.transform = self.transforms[split]
        self.allfiles = self.train_files if split == "train" else self.test_files
                
        self.label_id_mapping = set()
        
        self.all_files_paths = []
        self.all_labels = []
        
        for filename in self.allfiles["contactless"]:
            id = filename.split("/")[-1].split("_")[2] + filename.split("/")[-1].split("_")[4].lower() + filename.split("/")[-1].split("_")[-1].split(".")[0]
            self.label_id_mapping.add(id)
        
        for filename in self.allfiles["contactbased"]:
            id = filename.split("/")[-1].split("_")[1] + filename.split("/")[-1].split("_")[2].lower() + str(fingerdict[filename.split("/")[-1].split("_")[3].split(".")[0]])
            self.label_id_mapping.add(id)
            
        self.label_id_mapping = list(self.label_id_mapping)
        
        for filename in self.allfiles["contactless"]:
            id = filename.split("/")[-1].split("_")[2] + filename.split("/")[-1].split("_")[4].lower() + filename.split("/")[-1].split("_")[-1].split(".")[0]
            self.all_labels.append(self.label_id_mapping.index(id))
            self.all_files_paths.append(filename)
            
        for filename in self.allfiles["contactbased"]:
            id = filename.split("/")[-1].split("_")[1] + filename.split("/")[-1].split("_")[2].lower() + str(fingerdict[filename.split("/")[-1].split("_")[3].split(".")[0]])
            self.all_labels.append(self.label_id_mapping.index(id))
            self.all_files_paths.append(filename)
            
        print("Number of classes: ", len(self.label_id_mapping))
        print("Total number of images ", split ," : ", len(self.all_labels))
        
    def __len__(self):
        return len(self.all_files_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.all_files_paths[idx]

        sample = cv2.imread(filename)

        if self.transform:
            sample = self.transform(sample)

        return sample, self.all_labels[idx]
    
if __name__ == "__main__":
    ridgebase = RidgeBase(split = "train")
    dataloader = DataLoader(ridgebase, batch_size=4,
                        shuffle=True, num_workers=1)
    
    for image, label in dataloader:
        print(image.shape, label)