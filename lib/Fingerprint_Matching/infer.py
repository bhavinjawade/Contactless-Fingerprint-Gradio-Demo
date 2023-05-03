import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
from .train import Model
from tqdm import tqdm
import cv2
import torch.nn.functional as F

fingerdict = {
    "Index": 0,
    "Middle":1,
    "Ring": 2,
    "Little": 3
}

EVAL_MODEL='./lib/Fingerprint_Matching/ridgebase_NoTransform_LP2_NUMPOS3_BS64LR01_35.pt'
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model = Model(device).to(device)
ckpt = torch.load(EVAL_MODEL, map_location=torch.device('cpu'))
model.load_state_dict(ckpt)
model.eval()

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def match_fingerprints_cl2c(image1, image2):
    global model
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224,224)),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                        ])

    image1 = transform(image1)
    image2 = transform(image2)

    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)

    with torch.no_grad():
        image1, image2 = image1.to(device), image2.to(device)
        x_cl, x_cb = model(image1, image2)    
        print(x_cl.shape, x_cb.shape)
        sim = F.linear(l2_norm(x_cl), l2_norm(x_cb))
        return sim

def match_fingerprints_cl2cl(image1, image2):
    global model
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224,224)),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                        ])

    image1 = transform(image1)
    image2 = transform(image2)

    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)

    with torch.no_grad():
        image1, image2 = image1.to(device), image2.to(device)
        x_cl1, _ = model(image1, image2)    
        x_cl2, _ = model(image2, image1)    
        sim = F.linear(l2_norm(x_cl1), l2_norm(x_cl2))

        return sim

def match_full_cl2cl(segments1, segments2):
    global model
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224,224)),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                        ])
    all_sims = 0

    for i, segment1 in enumerate(segments1):
        segment2 = segments2[i]
        image1 = transform(segment1)
        image2 = transform(segment2)

        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)

        with torch.no_grad():
            image1, image2 = image1.to(device), image2.to(device)
            x_cl1, _ = model(image1, image2)    
            x_cl2, _ = model(image2, image1)    
            sim = F.linear(l2_norm(x_cl1), l2_norm(x_cl2))
            all_sims += sim
            print(sim)

    return all_sims / 4

def main_cl2c(fingerprint1, fingerprint2):
    sim = match_fingerprints_cl2c(fingerprint1, fingerprint2)
    score = sim.item()
    if (score > 0.35):
        pred = "Match Found"
    else:
        pred = "Not a Match"
    return sim.item(), pred

def main_cl2cl(fingerprint1, fingerprint2):
    sim = match_fingerprints_cl2cl(fingerprint1, fingerprint2)
    score = sim.item()
    if (score > 0.35):
        pred = "Match Found"
    else:
        pred = "Not a Match"
    return sim.item(), pred

def main_full_cl2cl(fingerprint1, fingerprint2):
    sim = match_full_cl2cl(fingerprint1, fingerprint2)
    score = sim.item()
    if (score > 0.50):
        pred = "Match Found"
    else:
        pred = "Not a Match"
    return sim.item(), pred
