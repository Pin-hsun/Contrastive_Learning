# Set up data loaders
from datasets import SiameseMNIST
from dataloader import MultiData, read_paired_path
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv

from Net3D2D import MRPretrained
from losses import ContrastiveLoss

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
parser.add_argument('--models', dest='models', type=str, help='use which models')
# Data
parser.add_argument('--dataset', type=str, default='default')
parser.add_argument('--preload', action='store_true')
parser.add_argument('--resize', type=int, default=0, help='size for resizing before cropping, 0 for no resizing')
parser.add_argument('--cropsize', type=int, default=256, help='size for cropping, 0 for no crop')
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
parser.add_argument('--load3d', action='store_true', dest='load3d', default=True, help='do 3D')
parser.add_argument('--trd', type=float, default=0)
parser.add_argument('--n01', action='store_true', dest='n01', default=False)
# Model
parser.add_argument('--backbone', type=str, default='resnet50', help='model backbone')
parser.add_argument('--pretrained', type=str, default=True, help='use pretrained model')
parser.add_argument('--n_classes', type=int, default=2, help='class number')
parser.add_argument('--fuse', type=str, default='cat', help='cat or max across the 2D slices')
# Training
parser.add_argument('-b', dest='batch_size', type=int, help='training batch size',default=2)
parser.add_argument('--n_epochs', type=int, help='# of iter at starting learning rate')
parser.add_argument('--lr', type=float, help='initial learning rate f -or adam')
parser.add_argument('--beta1', type=float, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--epoch_count', type=int, help='the starting epoch count')
parser.add_argument('--epoch_load', type=int, help='to load checkpoint form the epoch count')
parser.add_argument('--n_epochs_decay', type=int, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, help='learning rate policy: lambda|step|plateau|cosine')
# Loss
parser.add_argument('--lamb', type=int, help='weight on L1 term in objective')

def prepare_log(args):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    args.not_tracking_hparams = ['mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/', exist_ok=True)
    save_json(args, os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + '0.json')
    shutil.copy('models/' + args.models + '.py', os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + args.models + '.py')
    return args

def plot_embeddings(embeddings, targets, dest, xlim=None, ylim=None):
    plt.figure(figsize=(2,2))
    for i in range(2):
        # inds = np.where(targets==i)[0]
        # print(inds)
        # plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
        plt.scatter(embeddings[0], embeddings[1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.savefig(dest)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = []
        labels = []
        k = 0
        for images, target in dataloader:
            if cuda:
                images = [img.cuda() for img in images]
            out = [x.cpu().numpy() for x in model.forward(*images)]
            embeddings += out
            for i in target.numpy():
                labels.append(i)
            k += len(images)
    return embeddings, labels

# environment file
load_dotenv('env/.env')

# Read json file and update it
# with open('env/jsn/' + parser.parse_args().jsn + '.json', 'rt') as f:
#     t_args = argparse.Namespace()
#     t_args.__dict__.update(json.load(f)['train'])
#     args = parser.parse_args(namespace=t_args)


# Finalize Arguments and create files for logging
args = parser.parse_args()

# args = prepare_log(args)

# Set up the network and training parameters
margin = 1.
log_interval = 100
args.lr = 1e-2
args.n_epochs = 3

csv_path = 'data/test.csv'
root = '/media/ExtHDD02/OAIDataBase/'
img1_paths, img2_paths, labels = read_paired_path(csv_path)

train_set = MultiData(root=root, path=[img1_paths.tolist(), img2_paths.tolist()], labels=labels.tolist(),
                    opt=args, mode='train', filenames=False, index=range(7))
test_set = MultiData(root=root, path=[img1_paths.tolist(), img2_paths.tolist()], labels=labels.tolist(),
                    opt=args, mode='test', filenames=False, index=range(7,10))
print('train set:', train_set.__len__()) #467

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
loss_fn = ContrastiveLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# trainer = pl.Trainer(gpus=-1, strategy='ddp',
#                      max_epochs=args.n_epochs + 1, progress_bar_refresh_rate=20, logger=logger,
#                      enable_checkpointing=False)
# print(args)
# trainer.fit(model, train_loader, test_loader)  # test loader not used during training

if __name__ ==  '__main__':
    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, args.n_epochs, cuda, log_interval)
    train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_baseline, train_labels_baseline, dest='out/train_embedding.png')
    val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_baseline, val_labels_baseline, dest='out/test_embedding.png')

    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset siamese --prj 0105test --preload -b 2