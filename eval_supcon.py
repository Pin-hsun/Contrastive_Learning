import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.autograd import Variable

# other modules
import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
import argparse
import json
from dotenv import load_dotenv

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn import metrics
import statistics
import pickle
import seaborn as sns
from PIL import Image

# custom classes
from dataloader import MultiData, read_paired_path

parser = argparse.ArgumentParser()#add_help=False)
# Env
parser.add_argument('--jsn', type=str, default='default', help='name of ini file')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
# Project name
parser.add_argument('--prj', type=str, help='name of the project', default='test')
# Data
parser.add_argument('--dataset', type=str, default='default')
parser.add_argument('--resize', type=int, default=0, help='size for resizing before cropping, 0 for no resizing')
parser.add_argument('--cropsize', type=int, default=256, help='size for cropping, 0 for no crop')
parser.add_argument('--gray', action='store_true', dest='gray', help='dont copy img to 3 channel')
parser.add_argument('--load3d', action='store_true', dest='load3d', help='do 3D')
parser.add_argument('--trd', type=float, dest='trd', default=0, help='threshold of images')
parser.add_argument('--n01', dest='n01', action='store_true', help='normalize the image to 0~1')
parser.add_argument('--twocrop', action='store_true', dest='twocrop')
# Model
parser.add_argument('--epoch', type=str, default='default', help='use which epoch model')
# Step
parser.add_argument('--distance', action='store_true', help='calculate distance')
parser.add_argument('--plot', action='store_true', help='plot box plot')
parser.add_argument('--tsne', action='store_true', help='draw tsne plot')

def extract_embeddings(dataloader, model):
    model.eval()
    embeddings = []
    classes = []
    for data, target in dataloader:
        images = [img.cuda() for img in data]
        out = [x.detach().cpu().numpy() for x in model.forward(images)]
        embeddings += out
        target = [x.numpy() for x in target]
        classes += target
    embeddings = np.concatenate(embeddings)
    classes = np.concatenate(classes)

    return embeddings, classes

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# environment file
load_dotenv('env/.env')
# Read json file and update it
with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f)['test'])
    args = parser.parse_args(namespace=t_args)
print(args)

# loading siamese neural network model from the main.py output
output_folder_name = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints',
                                  'epoch' + str(args.epoch) +'.pth')
print(output_folder_name)
root = os.environ.get('DATASET')
net = torch.load(output_folder_name)
net.eval()

os.makedirs('plot/'+args.prj, exist_ok=True)

if args.tsne:
    # read test set path
    img_test, labels_test = read_paired_path('data/SupCon_pairs03_part_test.csv')
    test_set = MultiData(root=root, path=img_test, labels=labels_test, opt=args, mode='test', filenames=False)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False, pin_memory=True)
    X, y = extract_embeddings(test_loader, net)
    # n_samples, n_features = X.shape
    print(X.shape)
    # t-SNE
    X_tsne = TSNE(n_components=2, init='random', random_state=5, perplexity=30, n_iter=1000, verbose=1).fit_transform(X)
    # Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize
    plt.figure(figsize=(8, 8))
    color_dict = {0:'red', 1:'green'}
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=color_dict[y[i]], alpha=0.5)
    leg = plt.legend(color_dict.keys())
    for i, j in enumerate(leg.legendHandles):
        j.set_color(list(color_dict.values())[i])
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plot/' + args.prj + '/ep' + args.epoch + "_tsne.png")
    plt.close()

# CUDA_VISIBLE_DEVICES=0 python eval.py --prj 0117_cat_lre5 --epoch 17  --tsne --distance --plot