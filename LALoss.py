import copy
from tqdm import tqdm
import argparse
import os

import wandb

import numpy as np
import PIL
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

from models import ResNet
from dataset.longtail import sample, subsample, show_data_distribution, split
from utils.metrics import get_metrics

class LALoss(nn.Module):
    def __init__(self, prior, device):
        super(LALoss, self).__init__()
        self.device = torch.device(device)
        self.adjustment = torch.log(torch.tensor(np.array(prior)).to(self.device))
        self.adjustment.requires_grad = False
    def forward(self, inputs, targets, reduction='mean'):
        inputs = inputs + self.adjustment
        return F.cross_entropy(inputs, targets, reduction=reduction)



def train(trainloader, optimizer, net, criterion, epoch, device=torch.device('cuda:3'), separate_decay=False):
    '''
    Trains the model for one epoch
    ARGS:   
        trainloader: (torch dataloader)
        optimizer: torch optimizer
        net: torch model(Resnet56 here)
        criterion: Loss function
        epoch: training epoch
        device: cuda device with index or cpu to be used
        wandblogs: a dictionary of logs for wandb upload

    Returns a dictionary containing training logs
    '''
    running_loss = 0.0
    net.train()
    wandblogs={}
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        else:
            pass

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        if separate_decay:
            loss_r = 0
            for parameter in net.parameters():
                loss_r += torch.sum(parameter ** 2)
            loss = loss + 1e-4 * loss_r

        # zero the parameter gradients
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = (running_loss*i + loss.item())/(i + 1)
        wandblogs["train/running loss"] = running_loss
    return wandblogs


def feedforward(dataloader, net, device):
    '''
    Does a feed forward operation on all samples in the dataloder and returns
    the predictions and their corresponding labels
    '''
    net.eval()
    output_logs, label_logs = [], []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            else:
                pass
            labels = labels.cpu().detach().numpy()
            outputs = torch.argmax(net(inputs), dim=1).cpu().detach().numpy()
            output_logs.append(outputs)
            label_logs.append(labels)
        return (np.concatenate(output_logs, axis=0), np.concatenate(label_logs, axis=0))


def test(testloader, net, classes, device=torch.device('cuda:3')):
    '''
    testloader: a torch dataloader
    net: pytorch model
    classes: a list of strings of class names
    device: torch cuda/cpu device
    wandblog: a dictionary of logs of training

    returns a dictionary updated with test logs for the epoch
    '''
    predictions, labels = feedforward(testloader, net, device)
    wandblog = get_metrics(predictions, labels, classes)
    return wandblog


def loop(   trainloader, testloader, classes, prior, net, optimizer, epochs=1200, 
            lr_scheduler=None, device=torch.device('cuda:3'), separate_decay=False):
    logbar = tqdm(range(0, epochs), total=epochs, leave=False)
    max_acc = 0.0
    best_acc_model = None
    criterion = LALoss(prior, device)

    for i in logbar:
        wandblog = {}
        wandblog = wandblog|train(  trainloader, optimizer, net, criterion,
                                    epoch=i, device=device, separate_decay=separate_decay)
        lr_scheduler.step()

        wandblog = wandblog|test(testloader, net, classes, device)
        logbar.set_description(f"Epoch [{i}/{epochs}")

        logbar.set_description(f"Epoch [{i}/{epochs}")
        logbar.set_postfix(train_loss=f"{wandblog['train/running loss']:.2f}", val_acc=f"{wandblog['accuracy']:.2f}")

        if wandblog['accuracy'] > max_acc:
            best_acc_model = copy.deepcopy(net)
        wandb.log(wandblog)
    return best_acc_model


def param_group(model, args):
    param_group_list = [{'params':[], 'weight_decay':args.wdecay},
                        {'params':[], 'weight_decay':args.bdecay, 'momentum': 0.1}]
    for m in model.modules():
        if isinstance(m, nn.Linear):
            param_group_list[0]['params'].append(m.weight)
            param_group_list[0]['params'].append(m.bias)
        elif isinstance(m, nn.Conv2d):
            param_group_list[0]['params'].append(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            param_group_list[1]['params'].append(m.weight)
            param_group_list[1]['params'].append(m.bias)    
    return param_group_list


def parse():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--epochs', default=1200, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--max-steps', default=32, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--wdecay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--bdecay', default=0.9, type=float,
                        help='bnorm decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--wandb-project', default="LALossBaselines",
                        help='directory to output the result', type=str)
    parser.add_argument('--wandb-entity', default="stablegradients",
                        help='directory to output the result', type=str)
    parser.add_argument('--wandb-runid', default="maxmin_recall",
                        help='directory to output the result', type=str)
    parser.add_argument('--lt', type=bool, default=True,
                        help="don't use progress bae")
    parser.add_argument('--imbalance-ratio', type=float, default=100.0,
                        help="don't use progress bae")
    parser.add_argument('--savedir', default="./checkpoints/maxminrecall/", type=str,
                        help='directory to output the result')
    parser.add_argument('--arch', default="resnet", type=str,
                        help='directory to output the result')
    parser.add_argument('--separate-decay', default=False, type=bool)
    parser.add_argument('--opt-decay', default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = parse()
    print(args)
    wandb.init(project=args.wandb_project, id=args.wandb_runid, entity=args.wandb_entity)
    trainset, testset = sample(args.dataset)
    print("creating lt dataset")
    trainset, train_prior = subsample(trainset, args.imbalance_ratio)
    print("ther prior are ...")
    print(train_prior)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

   # get some random training images
    device = torch.device('cuda:' + str(args.gpu_id))
    num_classes=len(trainset.classes)
    if args.arch == 'resnet32':
        net = ResNet([(5, 16, 1), (5, 32, 2), (5, 64, 2)], num_classes).to(device)
    elif args.arch == 'resnet56':
         net = ResNet([(9, 16, 1), (9, 32, 2), (9, 64, 2)], num_classes).to(device)

    if args.arch == 'resnet32':
        net = ResNet([(5, 16, 1), (5, 32, 2), (5, 64, 2)], num_classes).to(device)
    elif args.arch == 'resnet56':
         net = ResNet([(9, 16, 1), (9, 32, 2), (9, 64, 2)], num_classes).to(device)

    if args.opt_decay:
        print("using only the opt decay params")
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay, nesterov=args.nesterov)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[600, 900, 1100], gamma = 0.1)
    else:
        print("using param groups values")
        param_group_list = param_group(net, args)
        optimizer = torch.optim.SGD(param_group_list, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[600, 900, 1100], gamma = 0.1)

    best_net = loop(trainloader, testloader, trainset.classes, train_prior, net, optimizer,
                    args.epochs, lr_scheduler, device=device, separate_decay=args.separate_decay)
    os.makedirs(args.savedir, exist_ok=True)
    torch.save(best_net.state_dict(), args.savedir + args.wandb_runid + ".pth")

main()