import numpy as np
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from pytorch_metric_learning import distances, losses, miners, reducers, testers
# from pytorch_metric_learning.utils import AccuracyCalculator
import pytorch_metric_learning.utils.logging_presets as logging_presets
from utils.SupCon_util import *

def offline_miner(b):
    # indices_tuple = (tensor([anc1, anc2]), tensor([pos1, pos2]), tensor([neg1, neg2]))
    anchor = torch.tensor([i for i in range(b)])*3
    pos = anchor+1
    neg = anchor+2
    indices_tuple = (anchor, pos, neg)
    return indices_tuple

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, device, train_loader, optimizer, epoch, checkpoints, writer):
    model.train()
    Loss = AverageMeter()

    tqdm_loader = tqdm(train_loader)
    for batch_idx, (data, labels) in enumerate(tqdm_loader):
        optimizer.zero_grad()
        labels = labels.view(labels.shape[0] * labels.shape[1])
        data = data.view(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4], data.shape[5])
        data, labels = data.to(device), labels.to(device)
        batch_size = data.shape[0]
        embeddings = model(data)  # torch.Size([4*B, out_feature])
        # indices_tuple = offline_miner(batch)
        # print(indices_tuple)
        # loss = loss_func(embeddings, labels, indices_tuple)
        loss = loss_func(embeddings, labels)
        loss = loss.mean()

        # back propagation
        loss.backward()
        optimizer.step()

        # visualize losses
        Loss.update(loss, batch_size)
        tqdm_loader.set_description('Train Epoch {ep} Loss {Loss.val:.4f} ({Loss.avg:.4f})  '.format(ep=epoch, Loss=Loss))

    writer.add_scalar('Loss/train', Loss.avg, epoch)
    #
    # if epoch % 1 == 0:
    #     torch.save(model, checkpoints + '/epoch' + str(epoch) + '.pth')

# def extract_embeddings(dataloader, model):
#     model.eval()
#     embeddings = []
#     classes = []
#     for data, target in dataloader:
#         images = [img.cuda() for img in data]
#         out = [x.detach().cpu().numpy() for x in model.forward(images)]
#         labels = [i.numpy() for i in target]
#         embeddings += out
#         classes += labels
#     classes = np.concatenate(classes)
#     embeddings = np.concatenate(embeddings)
#
#     return embeddings, classes

@torch.no_grad()
def test(model, loss_func, device, test_loader, epoch, writer):
    model.eval()
    Loss = AverageMeter()
    tqdm_loader = tqdm(test_loader)

    for batch_idx, (data, labels) in enumerate(tqdm_loader):
        labels = labels.view(labels.shape[0] * labels.shape[1])
        data = data.view(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4], data.shape[5])
        data, labels = data.to(device), labels.to(device)
        batch_size = data.shape[0]
        embeddings = model(data).detach().cpu()
        loss = loss_func(embeddings, labels)
        loss = loss.mean()
        
        # visualize losses
        Loss.update(loss, batch_size)
        tqdm_loader.set_description(
            'Test Epoch {ep} Loss {Loss.val:.4f} ({Loss.avg:.4f})  '.format(ep=epoch, Loss=Loss))
    writer.add_scalar('Loss/test', Loss.avg, epoch)
    return Loss.avg

if __name__ ==  '__main__':
    device = torch.device("cuda")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    batch_size = 12

    dataset1 = datasets.MNIST(".", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(".", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 1


    ### pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    # loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    loss_func = losses.NTXentLoss(temperature=0.5)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    mining_func = miners.EmbeddingsAlreadyPackagedAsTriplets()
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    ### pytorch-metric-learning stuff ###


    for epoch in range(1, num_epochs + 1):
        ori_train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        ori_test(dataset1, dataset2, model, accuracy_calculator)