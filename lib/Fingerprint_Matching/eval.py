import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
from datasets.ridgebase import RidgeBase
from utils import RetMetric
from train import Model
from tqdm import tqdm

# Paths for image directory and model
EVAL_MODEL='/home/bhavinja/Classification_Models/pytorch-image-classification/ridgebase_cnn_checkpoint_arcface28.pt'
# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(device).to(device)

# Load the model for evaluation
ckpt = torch.load(EVAL_MODEL)
model.load_state_dict(ckpt)

model.eval()

# Configure batch size and nuber of cpu's
num_cpu = multiprocessing.cpu_count()
bs = 8

test_kwargs = {'batch_size': 32, 'shuffle': False}
eval_dataset = RidgeBase(split="test")
eval_loader = torch.utils.data.DataLoader(eval_dataset, **test_kwargs)

# Evaluate the model accuracy on the dataset
all_feats = []
all_labels = []
with torch.no_grad():
    for images, target in tqdm(eval_loader):
        images, target = images.to(device), target.to(device)
        feat = model(images)
        feat = feat.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        all_feats.append(feat)
        all_labels.append(target)

    feats = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    print(feats.shape, labels.shape)
    retmetric = RetMetric(feats, labels)
    print("R@1: ", retmetric.recall_k(k=1))
    print("R@5: ", retmetric.recall_k(k=5))
    print("R@10: ", retmetric.recall_k(k=10))
   
