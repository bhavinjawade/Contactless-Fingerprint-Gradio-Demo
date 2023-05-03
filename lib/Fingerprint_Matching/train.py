from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from .datasets.ridgebase import RidgeBase
from .datasets.ridgebase_pair import RidgeBase_Pair
from .loss import DualMSLoss, get_Arcface, get_MSloss, get_ProxyAnchor
import timm
from .utils import RetMetric
from pprint import pprint
import numpy as np
from tqdm import tqdm
from .sampler import BalancedSampler
from torch.utils.data.sampler import BatchSampler

class Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.swin_cl = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True, num_classes=0, global_pool='')
        self.swin_cb = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True, num_classes=0, global_pool='')
        self.linear_cl = nn.Linear(1024, 1024)
        self.linear_cb = nn.Linear(1024, 1024)
        
        # self.swin_cl = timm.create_model('resnet50', pretrained=True, num_classes=0)
        # self.swin_cb = timm.create_model('resnet50', pretrained=True, num_classes=0)
        # self.linear_cl = nn.Linear(2048, 1024)
        # self.linear_cb = nn.Linear(2048, 1024)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1024)
        self.do = nn.Dropout(0.2)
        self.linear = nn.Linear(1024, 512)

    def forward(self, x_cl, x_cb):
        x_cl = self.swin_cl(x_cl)
        x_cl = x_cl.mean(dim=1)
        x_cl = self.linear_cl(x_cl)

        x_cb = self.swin_cb(x_cb)
        x_cb = x_cb.mean(dim=1)
        x_cb = self.linear_cb(x_cb)
        
        return x_cl, x_cb  

def train(args, model, device, train_loader, optimizers, epoch, loss_func):
    model.train()
    for batch_idx, (x_cl, x_cb, target) in enumerate(train_loader):
        x_cl, x_cb, target = x_cl.to(device), x_cb.to(device), target.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()
        x_cl, x_cb = model(x_cl, x_cb)
        loss = loss_func(x_cl, x_cb, target)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x_cl), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_feats = []
    all_labels = []
    print("Computing Test Recall")
    with torch.no_grad():
        for (x_cl, x_cb, target) in tqdm(test_loader):
            x_cl, x_cb, target = x_cl.to(device), x_cb.to(device), target.to(device)
            x_cl, x_cb = model(x_cl, x_cb)
            x_cb = x_cb.cpu().detach().numpy()
            x_cl = x_cl.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            all_feats.append(x_cl)
            all_feats.append(x_cb)
            all_labels.append(target)
            all_labels.append(target)
            
    feats = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    print(feats.shape, labels.shape)
    retmetric = RetMetric(feats, labels)
    print("R@1: ", retmetric.recall_k(k=1))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ccr', action='store_true', default=False,
                        help='enables training on CCR')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, default="swinmodel",
                        help='Name of the model for checkpointing')
    
    args = parser.parse_args()
            
    if (args.ccr):
        checkpoint_save_path = "/panasas/scratch/grp-doermann/bhavin/FingerPrintData/"
    else:
        checkpoint_save_path = "./"

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_dataset = RidgeBase_Pair(args.ccr, split="train")
    val_dataset = RidgeBase_Pair(args.ccr, split="test")
    
    balanced_sampler = BalancedSampler(train_dataset, batch_size=args.batch_size, images_per_class = 3)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args.batch_size, drop_last = True)
    
    train_kwargs = {'batch_sampler': batch_sampler}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    # model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    model = Model(device).to(device)
    
    optimizer_linear = optim.Adam(
        [
            {"params": model.linear_cl.parameters(), "lr":args.lr},
            {"params": model.linear_cb.parameters(), "lr":args.lr}
         ],
        lr=args.lr)
    
    optimizer_swin = optim.Adam(
        [
            {"params": model.swin_cl.parameters(), "lr":args.lr * 0.1},
            {"params": model.swin_cb.parameters(), "lr":args.lr * 0.1}
         ],
        lr=args.lr)
    
    # loss_func = get_Arcface(num_classes = 504, embedding_size = 1024)
    # loss_func = get_ProxyAnchor(num_classes = 504, embedding_size = 1024)
    # loss_func = get_MSloss()
    
    loss_func = DualMSLoss()
    
    scheduler_linear = StepLR(optimizer_linear, step_size=2, gamma=args.gamma)
    scheduler_swin = StepLR(optimizer_swin, step_size=2, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):            
        if (epoch > 5):
            print("Training with Swin")
            train(args, model, device, train_loader, [optimizer_linear, optimizer_swin], epoch, loss_func)
        else:
            print("Training only linear")
            train(args, model, device, train_loader, [optimizer_linear], epoch, loss_func)
        
        test(model, device, test_loader)
        
        if (epoch > 2):
            scheduler_linear.step()
            scheduler_swin.step()
        else:
            scheduler_linear.step()

        torch.save(model.state_dict(), checkpoint_save_path + "Models/ridgebase_" + args.model_name + "_" + str(epoch) + ".pt")


if __name__ == '__main__':
    main()