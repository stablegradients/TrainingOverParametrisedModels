import copy
import random
import math

import numpy as np

import matplotlib.pyplot as plt
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
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)

    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=test_transform)
    return trainset, testset


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


def subsample(dataset, imbalance=1.0):
    '''
    Subsample a given dataset with each class subsampled by a factor of λ^i
    based on a power law distribution. i is a random class index, i ∈ [number of classes]
    given a lambda
    ARGS:
        dataset         : a torch dataset with classes attr
        imbalance       : imbalance ratio
    RETURNS
        the dataset subsampled, the prior, and indices used for the power law function based
        subsampling
    '''
    dataset_class_wise = {}
    for i, name in enumerate(dataset.classes):
        dataset_class_wise[i] = []

    for i, (img, label) in enumerate(dataset):
        dataset_class_wise[label].append(i)
    
    for i, name in enumerate(dataset.classes):
        random.shuffle(dataset_class_wise[i])

    lamda = math.exp(-1 * math.log(imbalance)/(len(dataset.classes) - 1))
    for i, name in enumerate(dataset.classes):
        num_samples = max(int(lamda**i * len(dataset_class_wise[i])), 1)
        dataset_class_wise[i] = dataset_class_wise[i][:num_samples]

    select_list = []
    for i, name in enumerate(dataset.classes):
        select_list = select_list + dataset_class_wise[i]
    
    dataset.data=dataset.data[np.array(select_list)]
    dataset.targets=list(dataset.targets[x] for x in select_list)
    select_fraction = [lamda ** i  for i in range(len(dataset.classes))]

    prior = [x/sum(select_fraction) for x in select_fraction]
    return dataset, prior


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

