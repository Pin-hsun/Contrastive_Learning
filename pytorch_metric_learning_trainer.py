import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import pytorch_metric_learning.utils.logging_presets as logging_presets

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        labels = torch.cat((labels[1], labels[2])) # get pain label
        data = list(d.to(device) for d in data)
        labels = labels.to(device)
        optimizer.zero_grad()
        embeddings = model(*data)
        embeddings = torch.cat((embeddings[0], embeddings[1]), 0) #torch.Size([2, 69]
        # indices_tuple = mining_func(embeddings, labels)
        # loss = loss_func(embeddings, labels, indices_tuple)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )
            )

def ori_train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        # data shape: torch.Size([256, 1, 28, 28])
        # label: torch.Size([256])
        optimizer.zero_grad()
        embeddings = model(data) #torch.Size([256, 128])
        # indices_tuple = mining_func(embeddings, labels)
        # loss = loss_func(embeddings, labels, indices_tuple)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )
            )

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
        out = [x.detach().cpu().numpy() for x in model.forward(*images)]
        embeddings += out
        classes.append(target[1].numpy())
        classes.append(target[2].numpy())
    classes = np.concatenate(classes)
    embeddings = np.concatenate(embeddings)

    return embeddings, classes

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###

def test(train_laoder, test_laoder, model, accuracy_calculator):
    train_embeddings, train_labels = extract_embeddings(train_laoder, model)
    test_embeddings, test_labels = extract_embeddings(test_laoder, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        query=test_embeddings, query_labels=test_labels,
        reference=train_embeddings, reference_labels=train_labels,
        embeddings_come_from_same_source=False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

def ori_test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        query=test_embeddings, query_labels=test_labels,
        reference=train_embeddings, reference_labels=train_labels,
        embeddings_come_from_same_source=False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

if __name__ ==  '__main__':
    device = torch.device("cuda")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    batch_size = 256

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
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    ### pytorch-metric-learning stuff ###


    for epoch in range(1, num_epochs + 1):
        ori_train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        ori_test(dataset1, dataset2, model, accuracy_calculator)