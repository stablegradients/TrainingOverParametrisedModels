import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def sample(dataset="cifar10"):
    '''
    Returns dataloader and dataset for the trainset and testset respectively
    ARGS:
        dataset (str): either "cifar10" or "cifar100" for the corresponding dataset 
    '''
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[1.0, 1.0, 1.0])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    batch_size = 256
    assert dataset in ["cifar10", "cifar100"]

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=8)

    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=8)
    return trainloader, trainset, testloader, testset


import matplotlib.pyplot as plt
import numpy as np
import random


def show_data_distribution(dataset, keyname="no name"):
    '''
    Gives you a matplotlib plot of classwise distribution a given dataset
    ARGS:
        dataset: a torch dataset with classes attr
        keyname: whether trainset or testset, will be used in the plot heading
    '''
    classes = dataset.classes
    dataset_elems = len(classes)*[0]
    for image, label in dataset:
        dataset_elems[label]+=1
    plt.bar(list(classes), dataset_elems)
    plt.title(keyname)
    plt.show()


def subsample(dataset, lamda=1, class_indices=None):
    '''
    Subsample a given dataset with each class subsampled by a factor of λ^i
    based on a power law distribution. i is a random class index, i ∈ [number of classes]
    given a lambda
    ARGS:
        dataset         : a torch dataset with classes attr
        classes         : (int) num classes in the dataset
        lamda           : (float) the power law function base that is used to calculate 
                          the subsampling proportion
        class_indices   : (list) a list on integers, permute of list(range(number of classes))
                          if provided, used instead of the random index generation  
    RETURNS
        the dataset subsampled, the prior, and indices used for the power law function based
        subsampling
    '''
    num_classes = len(dataset.classes)
    if class_indices is None:
        class_indices = list(range(num_classes))
        random.shuffle(class_indices)
    
    class_probs = [lamda ** x for x in class_indices]
    prior = [x/sum(class_probs) for x in class_probs]
    select_list = []
    
    # subsample based on a biased cointoss
    for i, (img, label) in enumerate(dataset):
        if np.random.binomial(1, class_probs[label]):
            select_list.append(i)

    dataset.data=dataset.data[np.array(select_list)]
    dataset.targets=list(dataset.targets[x] for x in select_list)
    return dataset, prior, class_indices

def long_tail_test():
    trainloader, trainset, testloader, testset = sample()
    imbalanced_trainset, prior, indices = subsample(trainset, lamda = 0.7, class_indices=list(range(10)))
    imbalanced_testset, prior, indices = subsample(testset, lamda = 0.7, class_indices=list(range(10)))

    print("=" * 50)
    print("=" * 10 + "the trainset and the test set are exaclty prior matched here with a sequentially increasing class index" + "=" * 10)
    print("the per class priori is as follows")
    for i, elem in enumerate(imbalanced_trainset.classes):
        print(elem, " : " , round(prior[i],2))
    show_data_distribution(imbalanced_trainset, keyname="train dataset")
    show_data_distribution(imbalanced_testset, keyname="test dataset")

    trainloader, trainset, testloader, testset = sample()
    print("=" * 50)
    print("=" * 10 + " inverted train and test distribution"  + "=" * 10)
    print("=" * 10 + "the prior in trainset and the test set have the largest KL divergence loss here" + "=" * 10)
    imbalanced_trainset, prior, indices = subsample(trainset, lamda = 0.7)
    imbalanced_testset, prior, indices = subsample(testset, lamda = 0.7, class_indices=[9-x for x in indices])

    print("the per class priori is as follows")
    for i, elem in enumerate(imbalanced_trainset.classes):
        print(elem, " : " , round(prior[i],2))
    show_data_distribution(imbalanced_trainset, keyname="train dataset")
    show_data_distribution(imbalanced_testset, keyname="test dataset")
    return


import torch.nn as nn
import torch.nn.functional as F

class LALoss(nn.Module):
    def __init__(self, gain_matrix=np.eye(10)/10, device='cuda:3'):
        super(LALoss, self).__init__()
        self.device = torch.device(device)
        self.adjustment = torch.log(torch.tensor(np.diag(gain_matrix)).to(self.device))
        self.adjustment.requires_grad = False
    def forward(self, inputs, targets, reduction='mean'):
        inputs = inputs - self.adjustment
        return F.cross_entropy(inputs, targets, reduction=reduction)

import models


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def get_metrics(outputs, labels, classes):
    '''
    returns a dictionary of computed metrics
    ARGS
        outputs: (np.ndarray) a (N, # classes) dimensional array of output logits of the model
        labels: (np.ndarray) a (N) dimensional array where each element is the ground truth
                index of the corresponding output element
        classes: (list) a list of stings of names of classes
    RETURNS:
        a dictionary of classification metircs, support for:
        1. precision,
        2. recall,
        3. accuracy,
        4. max precision across all classes
        5. mean precision across all classes
        6. min precision  across all classes
        7. max recall  across all classes
        8. mean recall across all classes
        9. min recall  across all classes
        10. f1 micro average
        11. f1 macroa average

    '''
    precision = precision_score(labels, outputs, average=None, zero_division=0)
    precision_avg = precision_score(labels, outputs, average='macro', zero_division=0)
    max_precision = np.max(precision)
    min_precision = np.min(precision)
    mean_precision = np.mean(precision)
    
    recall = recall_score(labels, outputs, average=None, zero_division=0)
    recall_avg = recall_score(labels, outputs, average='macro', zero_division=0)
    max_recall = np.max(recall)
    min_recall = np.min(recall)
    mean_recall = np.mean(recall)
    
    f1_micro = f1_score(labels, outputs, average='micro')
    f1_macro = f1_score(labels, outputs, average='macro')

    accuracy = accuracy_score(labels, outputs)
    metrics =   {
                "precision": precision_avg,
                "recall": recall_avg,
                "accuracy": accuracy,
                "max_precision": max_precision,
                "mean_precision": mean_precision,
                "min_precision": min_precision,
                "max_recall": max_recall,
                "mean_recall": mean_recall,
                "min_recall": min_recall,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro
                }
    for i, name in enumerate(classes):
        metrics["precision_" + name] = precision[i]
        metrics["recall_" + name] = recall[i]

    return metrics, recall

def train(trainloader, optimizer, net, criterion, epoch, device=torch.device('cuda:3'), max_steps=50):
    '''
    TODO 
    '''
    running_loss = 0.0
    net.train()
    
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
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
        if i> max_steps:
            return running_loss
    return running_loss

def test(testloader, net, device=torch.device('cuda:3')):
    '''
    TODO
    '''
    net.eval()
    output_logs, label_logs = [], []

    for i, data in enumerate(testloader, 0):
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


from sklearn.metrics import confusion_matrix
import numpy as np

def validation(valloader, net, lamda, prior, classes, lr=0.1, device='cuda:3'):
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
    outputs, labels = test(valloader, net)
    metrics, recall = get_metrics(outputs, labels, classes)
    CM = confusion_matrix(labels, outputs, normalize="all")
    print(CM)
    lamda_ = [x * np.exp(-1 * lr * r) for x, r in zip(lamda, recall.tolist())]
    lamda_normalise  = sum(lamda)
    new_lamda = [x / lamda_normalise for x in lamda_]
    diagonal = [x/p for x,p in zip(new_lamda, prior)]
    return np.diag(diagonal), new_lamda, CM

from tqdm import tqdm
import copy



def loop(writer, trainloader, testloader, valloader, classes, net, optimizer, prior, lamda, epochs=1200, lr_scheduler=None, max_steps=50):
    logbar = tqdm(range(0, epochs), total=epochs, leave=False)
    max_acc = 0.0
    best_acc_model = None
    
    for i in logbar:
        if i<=600:
            lr = 0.1*2.71**(0.00765 * (i-600))
        else:
            lr = 0.1*2.71**(0.00765 * (600-i))
        G, lamda, CM = validation(valloader, net, lamda, prior, classes, lr=lr)
        criterion = LALoss(gain_matrix=G)
        train_loss = train(trainloader, optimizer, net, criterion, epoch=i, max_steps=max_steps)
        lr_scheduler.step()
        writer.add_scalar("train/loss", train_loss, i)

        outputs, labels = test(testloader, net)
        metrics, _ = get_metrics(outputs, labels, classes)
        for key in metrics.keys():
            writer.add_scalar("test/" + key, metrics[key], i)
        for j in range(10):
            writer.add_scalar("val/lambda - " + str(j + 1), lamda[j], i)
        logbar.set_description(f"Epoch [{i}/{epochs}")

        cm_diag = np.diag(CM).tolist()

        for j in range(10):
            writer.add_scalar("val/cm-diagonalBYprior - " + str(j + 1), cm_diag[j]/prior[j], i)
        
        for j in range(10):
            writer.add_scalar("val/cm-diagonal - " + str(j + 1), cm_diag[j], i)
        
        logbar.set_description(f"Epoch [{i}/{epochs}")
        acc = "accuracy"
        logbar.set_postfix(train_loss=f"{train_loss:.2f}", val_acc=f"{metrics[acc]:.2f}")
        if metrics[acc] > max_acc:
            best_acc_model = copy.deepcopy(net)
    return best_acc_model

import random


def split(dataset):
    dataset_len = len(dataset)
    splits = int(0.2 * dataset_len )
    indices = list(range(dataset_len))
    random.shuffle(indices)
    
    val_idx, train_idx = indices[:splits], indices[splits:]
    valset, trainset = copy.deepcopy(dataset), copy.deepcopy(dataset)
    
    valset.data=dataset.data[np.array(val_idx)]
    valset.targets=list(dataset.targets[x] for x in val_idx)

    trainset.data=dataset.data[np.array(train_idx)]
    trainset.targets=list(dataset.targets[x] for x in train_idx)
    return valset, trainset


import torchvision.transforms as transforms

import PIL

from torchvision.models.resnet import _resnet
import torchvision



for lamda in [0.6]:
    writer = SummaryWriter(log_dir="./test")
    batch_size = 128
    trainloader, trainset, testloader, testset = sample()
    trainset, train_prior, indices = subsample(trainset, lamda, class_indices=list(range(len(trainset.classes))))
    valset, trainset = split(trainset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)

    show_data_distribution(trainset, keyname="train dataset")
    show_data_distribution(testset, keyname="test dataset")
    show_data_distribution(valset, keyname="val dataset")

   # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    device = torch.device('cuda:3')
    net = models.resnet32(num_classes=10).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[600, 900, 1000], gamma = 0.1)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    def runif_in_simplex(n):
        ''' Return uniformly random vector in the n-simplex '''

        k = np.random.randint(100, size=10)
        l = k / sum(k)
        l = l.tolist()
        random.shuffle(l)
        return l
    best_net = loop(writer, trainloader, testloader, valloader, trainset.classes ,net, optimizer, train_prior, [0.1]*10 , 1200, lr_scheduler, max_steps=50)
    torch.save(best_net.state_dict(), "./test")