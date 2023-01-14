import argparse
import json
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt

classes = ['0', '1']
colors = ['#1f77b4', '#ff7f0e']

def plot_embeddings(embeddings, classes, dest, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(len(embeddings)):
        if classes[i] == 0:
            plt.scatter(embeddings[i][0][0], embeddings[i][0][1], alpha=0.5, color=colors[0])
        else:
            plt.scatter(embeddings[i][0][0], embeddings[i][0][1], alpha=0.5, color=colors[1])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    # plt.legend(classes)
    plt.savefig(dest)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = []
        classes = []
        k = 0
        for images, target in dataloader:
            if cuda:
                images = [img.cuda() for img in images]
            out = [x.cpu().numpy() for x in model.forward(*images)]
            embeddings += out
            classes.append(target[1].numpy())
            classes.append(target[2].numpy())
            k += len(images)
    return embeddings, classes

# Set up data loaders
batch_size = 256
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

if __name__ ==  '__main__':
    os.makedirs(os.path.join('out', args.prj), exist_ok=True)
    for epoch in range(*args.nepochs):

        train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
        plot_embeddings(train_embeddings_baseline, train_labels_baseline, dest=os.path.join('out', args.prj, 'train_embedding.png'))
        val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
        plot_embeddings(val_embeddings_baseline, val_labels_baseline, dest=os.path.join('out', args.prj, 'test_embedding.png'))
