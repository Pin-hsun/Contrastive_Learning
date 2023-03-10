import pandas as pd
from dataloader import MultiData, read_paired_path
from models.Net3D2D import MRPretrained
from losses import ContrastiveLoss
from pytorch_metric_learning import losses
from utils.make_config import save_json
from trainer import fit

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
import os
import time
import shutil
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


classes = ['0', '1']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

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
parser.add_argument('--part_data', action='store_true', help='run partial data for scrip testing')
# Model
parser.add_argument('--model', type=str, default='Net3D2D', help='model name')
parser.add_argument('--backbone', type=str, default='resnet50', help='model backbone')
parser.add_argument('--pretrained', type=bool, help='use pretrained model')
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

def read_paired_path(csv_path):
    # path_list = []
    df = pd.read_csv(csv_path)
    paths = df.loc[df.columns.str.startswith('path')]
    path_list = [i.tolist() for i in paths]
    # img1_path = df['path1'].tolist()
    # img2_path = df['path2'].tolist()
    # labels = df[['label', 'painL', 'painR']]
    # labels = list(zip(df.label, df.painL, df.painR))
    labels = list(zip(df.label, df.V00WOMKPL.astype('int32'), df.V00WOMKPR.astype('int32'))) #pytorch metric learning
    return img1_path, img2_path, labels

def prepare_log(args):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    args.not_tracking_hparams = ['mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/', exist_ok=True)
    save_json(args, os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + '0.json')
    shutil.copy(args.model + '.py', os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + args.model + '.py')
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
    for data, target in train_loader:
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

# read csv file to get img path
if args.part_data:
    train_csv = 'data/part_train.csv'
    test_csv = 'data/part_test.csv'
else:
    train_csv = 'data/womac_pairs05_train.csv'
    test_csv = 'data/womac_pairs05_test.csv'
root = os.environ.get('DATASET')
img1_train, img2_train, labels_train = read_paired_path(train_csv)
img1_test, img2_test, labels_test = read_paired_path(test_csv)
# train_index, test_index = train_test_split(list(range(len(labels))), test_size=0.3, random_state=42)
# test = pd.read_csv(csv_path).iloc[test_index]
# test.to_csv()

train_set = MultiData(root=root, path=[img1_train, img2_train], labels=labels_train,
                    opt=args, mode='train', filenames=False)
test_set = MultiData(root=root, path=[img1_test, img2_test], labels=labels_test,
                    opt=args, mode='test', filenames=False)
print('train set:', train_set.__len__()) #467
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

# Trainer
checkpoints = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints')
os.makedirs(checkpoints, exist_ok=True)
model = MRPretrained(args_m=args)
model.cuda()
loss_fn = ContrastiveLoss(args.margin)
# loss_fn = losses.NTXentLoss(temperature=0.07)
if args.op == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.op == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

# trainer = pl.Trainer(gpus=-1, strategy='ddp',
#                      max_epochs=args.n_epochs + 1, progress_bar_refresh_rate=20, logger=logger,
#                      enable_checkpointing=False)
# trainer.fit(model, train_loader, test_loader)  # test loader not used during training

if __name__ ==  '__main__':
    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, args.n_epochs, cuda, args.log_interval, checkpoints, args.prj)
    # plot embedding outcomes
    # os.makedirs(os.path.join('out', args.prj), exist_ok=True)
    # train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model, args.fc_use)
    # plot_embeddings(train_embeddings_baseline, train_labels_baseline, dest=os.path.join('out', args.prj, 'train_embedding.png'))
    # val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model, args.fc_use)
    # plot_embeddings(val_embeddings_baseline, val_labels_baseline, dest=os.path.join('out', args.prj, 'test_embedding.png'))

    # CUDA_VISIBLE_DEVICES=1 python main.py --prj 0117_cat_lre5 --lr 0.00001