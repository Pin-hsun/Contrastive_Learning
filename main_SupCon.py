import pandas as pd
import numpy as np
from dataloader import PairedData3D, read_paired_path
from models.new_net import MRPretrained
from losses import ContrastiveLoss
from pytorch_metric_learning import losses
from utils.make_config import save_json

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
import os
import time
import shutil
from tqdm import tqdm
from dotenv import load_dotenv

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from trainers.trainer_pytorch_metric_learning import train, test

# Arguments
parser = argparse.ArgumentParser()#add_help=False)
# Env
parser.add_argument('--jsn', type=str, default='default', help='name of ini file')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
# Project name
parser.add_argument('--prj', type=str, help='name of the project', default='test')
# Data
parser.add_argument('--dataset', type=str, default='default')
parser.add_argument('--preload', action='store_true')
parser.add_argument('--resize', type=int, default=0, help='size for resizing before cropping, 0 for no resizing')
parser.add_argument('--cropsize', type=int, default=256, help='size for cropping, 0 for no crop')
parser.add_argument('--gray', action='store_true', dest='gray', help='dont copy img to 3 channel')
parser.add_argument('--load3d', action='store_true', dest='load3d', help='do 3D')
parser.add_argument('--trd', type=float, dest='trd', default=0, help='threshold of images')
parser.add_argument('--n01', dest='n01', action='store_true', help='normalize the image to 0~1')
parser.add_argument('--twocrop', action='store_true', dest='twocrop')
parser.add_argument('--part_data', action='store_true', help='run partial data for scrip testing')
# Model
parser.add_argument('--model', type=str, default='new_net', help='model name')
parser.add_argument('--backbone', type=str, default='resnet50', help='model backbone')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--n_classes', type=int, default=2, help='class number')
parser.add_argument('--fuse', type=str, default='cat', help='cat or max across the 2D slices')
parser.add_argument('--op', dest='optimizer', type=str, help='adam or sgd')
parser.add_argument('--fc_use', action='store_true', help='use fc in model last layer')
# Training
parser.add_argument('-b', dest='batch_size', type=int, help='training batch size')
parser.add_argument('--n_epochs', type=int, help='# of iter at starting learning rate')
parser.add_argument('--lr', type=float, help='initial learning rate f -or adam')
parser.add_argument('--beta1', type=float, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, help='number of threads for data loader to use')
parser.add_argument('--epoch_count', type=int, help='the starting epoch count')
parser.add_argument('--epoch_load', type=int, help='to load checkpoint form the epoch count')
parser.add_argument('--n_epochs_decay', type=int, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--log_interval', type=int, help='show loos of n interval')
parser.add_argument('--margin', type=int, help='greater than some margin value if they represent different classes')

def prepare_log(args):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    args.not_tracking_hparams = ['mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/', exist_ok=True)
    save_json(args, os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + '0.json')
    shutil.copy('models/'+ args.model + '.py', os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + args.model + '.py')
    return args

def plot_embeddings(embeddings, classes, dest, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(len(embeddings)):
        plt.scatter(embeddings[i][0], embeddings[i][1], alpha=0.5, color=colors[classes[i]])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    # plt.legend(classes)
    plt.savefig(dest)

def extract_embeddings(dataloader, model, fc_use):
    model.eval()
    embeddings = []
    classes = []
    for data, target in dataloader:
        images = [img.cuda() for img in data]
        if fc_use:
            out = [x.detach().cpu().numpy() for x in model.forward(*images)]
        else:
            out = [model.fc(x).detach().cpu().numpy() for x in model.forward(*images)]
        embeddings += out
        classes.append(target[1].numpy())
        classes.append(target[2].numpy())
    classes = np.concatenate(classes)
    embeddings = np.concatenate(embeddings)

    return embeddings, classes

# environment file
load_dotenv('env/.env')

# Read json file and update it
with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f)['train'])
    args = parser.parse_args(namespace=t_args)

# Finalize Arguments and create files for logging
args = prepare_log(args)
print(args)
cuda = torch.cuda.is_available()

# read csv file to get img path
if args.part_data:
    train_csv = 'data/SupCon_pairs03_part_train.csv'
    test_csv = 'data/SupCon_pairs03_part_test.csv'
else:
    train_csv = 'data/SupCon_pairs03_train.csv'
    test_csv = 'data/SupCon_pairs03_test.csv'
root = os.environ.get('DATASET')
img_train, labels_train = read_paired_path(train_csv, 300)
img_test, labels_test = read_paired_path(test_csv, 100)

# # construct data loader for 'cifar100'
# mean = (0.5071, 0.4867, 0.4408)
# std = (0.2675, 0.2565, 0.2761)
# normalize = transforms.Normalize(mean=mean, std=std)
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(size=args.cropsize, scale=(0.2, 1.)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     normalize,
# ])

train_set = PairedData3D(root=root, paths=img_train, labels=labels_train,
                    opt=args, mode='train', filenames=False)
test_set = PairedData3D(root=root, paths=img_test, labels=labels_test,
                    opt=args, mode='test', filenames=False)
print('train set:', train_set.__len__())
print('test set:', test_set.__len__())

train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=False, pin_memory=True)
# preload
if args.preload:
    tini = time.time()
    print('Preloading...')
    for i, x in enumerate(tqdm(train_loader)):
        pass
    if test_loader is not None:
        for i, x in enumerate(tqdm(test_loader)):
            pass
    print('Preloading time: ' + str(time.time() - tini))

# Logger
logger = pl_loggers.TensorBoardLogger(os.environ.get('LOGS') + args.dataset + '/', name=args.prj)
writer = SummaryWriter('./outs/' + args.prj)
# Trainer
checkpoints = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints')
os.makedirs(checkpoints, exist_ok=True)
model = MRPretrained(args_m=args)
model.cuda()
if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

### pytorch-metric-learning stuff ###
device = torch.device("cuda")
# distance = distances.CosineSimilarity()
# loss_func = losses.SupConLoss(temperature=0.1)
loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
# mining_func = miners.TripletMarginMiner(
# mining_func = miners.TripletMarginMiner(
#     margin=0.2, distance=distance, type_of_triplets="semihard"
# )
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

for epoch in range(1, args.n_epochs+1):
    train(model, loss_func, device, train_loader, optimizer, epoch, checkpoints, writer)
    # test(model, loss_func, device, test_loader, epoch, writer)

#  CUDA_VISIBLE_DEVICES=2 python main_SupCon.py --prj 0216_test --twocrop -b 16 --n_epochs 30 --lr 0.00001 --part_data