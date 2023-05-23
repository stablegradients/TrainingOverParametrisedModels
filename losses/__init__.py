import numpy as np
from operator import truediv

import torch
import torch.nn.functional as F


class HybridLoss(torch.nn.Module):
    def __init__(self, device='cuda:0'):
        super(HybridLoss, self).__init__()
        self.device = device

    def gain(self, G):
        D = np.diag(np.diag(G))
        M = np.matmul(G, np.linalg.inv(D))

        self.G = torch.tensor(G, requires_grad=False).to(self.device)
        self.M = torch.tensor(M, requires_grad=False).to(self.device)
        self.D = torch.tensor(D, requires_grad=False).to(self.device)
        self.adjustment = torch.log(torch.tensor(np.diag(D), requires_grad=False).to(self.device))

    def forward(self, inputs, targets, reduction='mean'):
        log_probs = F.log_softmax(inputs - self.adjustment , dim=1)
        weights = self.M[targets.cpu()].to(torch.device(self.device))
        product = weights * log_probs
        if reduction=='mean':
            return -1 * torch.mean(torch.sum(product, 1))
        else:
            return -1 * torch.sum(product, 1)

class MinRecall(HybridLoss):
    def __init__(self, prior, val_lr, device='cuda:0'):
        super(MinRecall, self).__init__(device)
        self.prior = prior
        self.num_classes = len(self.prior)
        self.lambdas = [0.1] * self.num_classes
        self.val_lr = val_lr
    
    def update(self, CM):
        tp = np.diag(CM)
        recall = list(map(truediv, tp, np.sum(CM, axis=1)))
        new_lambdas = [x * np.exp(-1 * self.val_lr * r) for\
                      x, r in zip(self.lambdas, recall)]
        new_lambdas = [x/sum(new_lambdas) for x in new_lambdas]
        self.lambdas = new_lambdas
        diagonal = [x/p for x,p in zip(self.lambdas, self.prior)]
        G = np.diag(diagonal)
        self.gain(G)
        return

class MinHTRecall(HybridLoss):
    def __init__(self, prior, val_lr, device='cuda:0'):
        super(MinHTRecall, self).__init__(device)
        self.prior = prior
        self.num_classes = len(self.prior)
        self.lambdas = [0.1] * self.num_classes
        self.val_lr = val_lr
    
    def update(self, CM):
        tp = np.diag(CM)
        recall = list(map(truediv, tp, np.sum(CM, axis=1)))
        num_classes = len(recall)
        head_recall = np.mean(np.array(recall[:int(0.9 * num_classes)]))
        tail_recall = np.mean(np.array(recall[int(0.9 * num_classes):]))

        lambda_h, lambda_t = self.lambdas[0], self.lambdas[-1]

        lambda_h = lambda_h * np.exp(-1 * self.val_lr * head_recall)
        lambda_t = lambda_t * np.exp(-1 * self.val_lr * tail_recall)

        lambda_h = lambda_h/(lambda_h + lambda_t)
        lambda_t = 1 - lambda_h
        self.lambdas = [lambda_h] * int(0.9 * num_classes) + [lambda_t] * int(0.1 * num_classes)
        diagonal = []
        for i in range(self.num_classes):
            if i < 0.9 * self.num_classes:
                diagonal.append(lambda_h/0.9)
            else:
                diagonal.append(lambda_t/0.1)
        G = np.diag(diagonal)
        self.gain(G)
        return


class MeanRecallCoverage(HybridLoss):
    def __init__(self, prior, val_lr, device='cuda:0'):
        super(MeanRecallCoverage, self).__init__(device)
        self.prior = prior
        self.num_classes = len(self.prior)
        self.lambdas = [0.0] * self.num_classes
        self.val_lr = val_lr
    
    def update(self, CM):
        tp = np.diag(CM)
        recall = list(map(truediv, tp, np.sum(CM, axis=1)))
        new_lamdas = []
        C = np.sum(CM, axis=0).tolist()
        for i, (l, c, p) in enumerate(zip(self.lambdas, C, self.prior)):
            l_ = l - self.val_lr * (c - 0.95/self.num_classes)
            l_ = max(0, l_)
            new_lamdas.append(l_)
        self.lambdas = new_lamdas
        G = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i==j:
                    G[i, i] = 1.0/self.prior[i] + new_lamdas[i]
                else:
                    G[i, j] = new_lamdas[j]
        self.gain(G)
        return


class MeanRecallHTCoverage(HybridLoss):
    def __init__(self, prior, val_lr, device='cuda:0'):
        super(MeanRecallHTCoverage, self).__init__(device)
        self.prior = prior
        self.num_classes = len(self.prior)
        self.lambdas = [0.0] * self.num_classes
        self.val_lr = val_lr

    def update(self, CM):
        C = np.sum(CM, axis=0).tolist()

        l_head, l_tail = self.lambdas[0], self.lambdas[-1]
        head_coverage, tail_coverage =  np.mean(C[:int(0.9 * self.num_classes)]),\
                                        np.mean(C[int(0.9 * self.num_classes):]) 
        l_head = l_head - self.val_lr * (head_coverage - 0.95/self.num_classes)
        l_tail = l_tail - self.val_lr * (tail_coverage - 0.95/self.num_classes)
        l_head = max(0, l_head)
        l_tail = max(0, l_tail)
        new_lamdas = [l_head] * int(0.9 * self.num_classes) + [l_tail] * int(0.1 * self.num_classes)
        self.lambdas = [l_head] * int(0.9 * self.num_classes) + [l_tail] * int(0.1 * self.num_classes)
        G = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i==j:
                    G[i, i] = (1.0/self.prior[i]) + new_lamdas[j]/int(0.9 * self.num_classes)
                else:
                    G[i, j] = new_lamdas[j]/int(0.1 * self.num_classes)
        self.gain(G)
        return
