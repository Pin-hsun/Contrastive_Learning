import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import pytorch_metric_learning.utils.logging_presets as logging_presets

def offline_miner(b):
    # indices_tuple = (tensor([anc1, anc2]), tensor([pos1, pos2]), tensor([neg1, neg2]))
    anchor = torch.tensor([i for i in range(b)])*3
    pos = anchor+1
    neg = anchor+2
    indices_tuple = (anchor, pos, neg)
    return indices_tuple

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, device, train_loader, test_loader, optimizer, epoch, checkpoints, writer):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # labels = torch.cat((labels[1], labels[2])) # get pain label
        labels = torch.cat(labels).to(device)
        # data = torch.cat(data).to(device)
        data = list(d.to(device) for d in data)
        # labels = list(d.to(device) for d in labels)
        optimizer.zero_grad()
        embeddings = model(data)
        embeddings = torch.cat(embeddings, 0) #torch.Size([4*B, fc_out_feat]
        # indices_tuple = offline_miner(batch)
        # print(indices_tuple)
        # loss = loss_func(embeddings, labels, indices_tuple)
        loss = loss_func(embeddings, labels)
        train_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )
            )
    train_loss /= (batch_idx + 1)
    writer.add_scalar('Loss/train', train_loss, epoch)
    model.eval()
    val_loss = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        labels = torch.cat(labels)
        data = list(d.to(device) for d in data)
        labels = labels.to(device)
        embeddings = model(data)
        embeddings = torch.cat(embeddings, 0)
        loss = loss_func(embeddings, labels)
        val_loss += loss

    val_loss /= (batch_idx + 1)
    writer.add_scalar('Loss/test', val_loss, epoch)
    if epoch % 1 == 0:
        torch.save(model, checkpoints + '/epoch' + str(epoch) + '.pth')

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def extract_embeddings(dataloader, model):
    model.eval()
    embeddings = []
    classes = []
    for data, target in dataloader:
        images = [img.cuda() for img in data]
        out = [x.detach().cpu().numpy() for x in model.forward(images)]
        labels = [i.numpy() for i in target]
        embeddings += out
        classes += labels
    classes = np.concatenate(classes)
    embeddings = np.concatenate(embeddings)

    return embeddings, classes

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###

def test(model, train_loader, test_loader, accuracy_calculator):
    train_embeddings, train_labels = extract_embeddings(train_loader, model)
    test_embeddings, test_labels = extract_embeddings(test_loader, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        query=test_embeddings, query_labels=test_labels,
        reference=train_embeddings, reference_labels=train_labels,
        embeddings_come_from_same_source=False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

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