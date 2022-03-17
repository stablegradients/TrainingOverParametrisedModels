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

import models
from dataset.longtail import sample, subsample, show_data_distribution, split
from utils.metrics import get_metrics

from resnet import ResNet, resnet56, BasicBlock

class LALoss(nn.Module):
    def __init__(self, gain_matrix, device):
        super(LALoss, self).__init__()
        self.device = torch.device(device)
        self.adjustment = torch.log(torch.tensor(np.diag(gain_matrix)).to(self.device))
        self.adjustment.requires_grad = False
    def forward(self, inputs, targets, reduction='mean'):
        inputs = inputs - self.adjustment
        return F.cross_entropy(inputs, targets, reduction=reduction)


def train(trainloader, optimizer, net, criterion, epoch, device=torch.device('cuda:3'), max_steps=50):
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

        '''
        the following regularization is something Aditya Krishna Menon, the co-author
        of both Training Overparametrised models with non decomposable objectives 
        and Logit adjustment for Long tail Learning did in his code for LA
        '''
        loss_r = 0
        for parameter in net.parameters():
            loss_r += torch.sum(parameter ** 2)
        loss = loss + 1e-3 * loss_r

        # zero the parameter gradients
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = (running_loss*i + loss.item())/(i + 1)
        wandblogs["train/running loss"] = running_loss
        if i> max_steps:
            return wandblogs
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
    num_classes = len(classes)
    wandblog = get_metrics(predictions, labels, classes)
    return wandblog


def validation( valloader, net, lamda, prior, wandblog={}, val_lr=0.1, device=torch.device('cuda:3')):
    '''
    Args:
        valloader: a torch dataloader
        net: classifier model
        lamda: a list of lagrange multiliers float
        prior: list of float priors.
        lr: learning rate for ED technique
    Returns:
        the gain matrix and lamda
    '''
    net.eval()
    num_classes = len(prior)
    outputs, labels = feedforward(valloader, net, device)
    CM = confusion_matrix(labels, outputs, normalize="all")
    recall = recall_score(labels, outputs, average=None, zero_division=0)
    head_recall = np.mean(recall[:int(0.9 * num_classes)])
    tail_recall = np.mean(recall[int(0.9 * num_classes):])

    lamda_h, lamda_t = lamda[0], lamda[-1]

    lamda_h = lamda_h * np.exp(-1 * val_lr * head_recall)
    lamda_t = lamda_t * np.exp(-1 * val_lr * tail_recall)

    lamda_h = lamda_h/(lamda_h + lamda_t)
    lamda_t = 1 - lamda_h

    new_lamdas_ = [lamda_h/(0.9 * num_classes)] * int(0.9 * num_classes) + [lamda_t/(0.1 * num_classes)] * int(0.1 * num_classes)
    new_lamdas = [lamda_h] * int(0.9 * num_classes) + [lamda_t] * int(0.1 * num_classes)

    for i, l in enumerate(new_lamdas):
        wandblog["val/lambda " + str(i)] = l

    diagonal = [x/p for x,p in zip(new_lamdas_, prior)]
    G = np.diag(diagonal)

    return G, new_lamdas, CM, wandblog


def loop(   trainloader, testloader, valloader, val_lr,
            classes, net, optimizer, prior, lamda, epochs=1200, 
            lr_scheduler=None, max_steps=50, device=torch.device('cuda:3'), step_lr_val=False, inc_steps=False):
    logbar = tqdm(range(0, epochs), total=epochs, leave=False)
    max_acc = 0.0
    best_acc_model = None
    for i in logbar:
        wandblog = {}
        '''
        if i== or i==90:
            for param in optimizer.param_groups:
                param['weight_decay'] = 0.1
        else:
            for param in optimizer.param_groups:
                param['weight_decay'] = 0.1
        '''
        if step_lr_val:
            if i in [600, 900, 1000]:
                val_lr = val_lr * 0.1
        if inc_steps and i > 600:
            max_steps = 90

        G, lamda, CM, wandblog = validation(valloader=valloader, net=net, lamda=lamda,
                                            prior=prior, wandblog=wandblog, val_lr=val_lr, device=device)
        criterion = LALoss(gain_matrix=G, device=device)

        wandblog = wandblog|train(trainloader, optimizer, net, criterion, epoch=i, max_steps=max_steps, device=device)
        lr_scheduler.step()

        wandblog = wandblog|test(testloader, net, classes, device)
        logbar.set_description(f"Epoch [{i}/{epochs}")

        logbar.set_description(f"Epoch [{i}/{epochs}")
        logbar.set_postfix(train_loss=f"{wandblog['train/running loss']:.2f}", val_acc=f"{wandblog['accuracy']:.2f}")

        if wandblog['accuracy'] > max_acc:
            best_acc_model = copy.deepcopy(net)
        wandb.log(wandblog)
    return best_acc_model


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
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--vlr', '--validation-learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--wdecay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--bdecay', default=0.9, type=float,
                        help='bnorm decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--wandb-project', default="MaxMinHTRecall",
                        help='directory to output the result', type=str)
    parser.add_argument('--wandb-entity', default="stablegradients",
                        help='directory to output the result', type=str)
    parser.add_argument('--wandb-runid', default="maxmin_recall",
                        help='directory to output the result', type=str)
    parser.add_argument('--lt', type=bool, default=True,
                        help="don't use progress bae")
    parser.add_argument('--nestrov', type=bool, default=True,
                        help="don't use progress bae")
    parser.add_argument('--imbalance-ratio', type=float, default=100.0,
                        help="don't use progress bae")
    parser.add_argument('--savedir', default="./checkpoints/maxminrecall/", type=str,
                        help='directory to output the result')
    parser.add_argument('--arch', default="resnet", type=str,
                        help='directory to output the result')
    parser.add_argument('--dual-norm', default=False, type=bool)
    parser.add_argument('--step-val', default=False, type=bool)
    parser.add_argument('--inc-steps', default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = parse()
    print(args)
    wandb.init(project=args.wandb_project, id=args.wandb_runid, entity=args.wandb_entity)
    trainset, testset = sample(args.dataset)
    if args.lt:
        print("creating lt dataset")
        trainset, train_prior = subsample(trainset, args.imbalance_ratio)
    else:
        train_prior = [1.0/len(trainset.classes)] * len(trainset.classes)

    valset, testset = split(testset, split_size=0.5)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

   # get some random training images
    device = torch.device('cuda:' + str(args.gpu_id))
    num_classes=len(trainset.classes)
    net = ResNet(BasicBlock, [9, 9, 9], num_classes).to(device)

    if args.dual_norm:
        wd_params = set()
        bd_params = set()
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                wd_params.add(m.weight)
            if isinstance(m, (nn.BatchNorm2d)):
                bd_params.add(m.bias)
        opt_list = [{'params': list(wd_params), 'weight_decay': args.wdecay},
                    {'params': list(bd_params), 'weight_decay': args.bdecay}]
        optimizer = torch.optim.SGD(opt_list, lr=args.lr, momentum=0.9, nesterov=args.nestrov)
    
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[600, 900, 1000], gamma = 0.1)

    lamda_init = [1.0/num_classes] * num_classes
    best_net = loop(trainloader, testloader, valloader, args.vlr, trainset.classes ,net, optimizer,
                    train_prior, lamda_init, args.epochs, lr_scheduler, max_steps=args.max_steps, device=device, step_lr_val=args.step_val, inc_steps=args.inc_steps)
    os.makedirs(args.savedir, exist_ok=True)
    torch.save(best_net.state_dict(), args.savedir + args.wandb_runid + ".pth")

main()