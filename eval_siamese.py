'''
Siamese ROP evaluation
created 5/21/2019
Part 1:
- Analysis of 100 excluded samples from the dataset, previously annotated by experts for disease severity ranking
- Show that we can learn the severity of clinical grade to a finer degree of continuous variation than just normal, pre-plus, and plus
- Euclidean distance from siamese network model should reflect these distances
Part 2:
- Analysis of test set paired image comparison median Euclidean distance relative to a randomly sampled pool of normal images
Part 3:
- Analysis of longitudinal change in disease severity in the test set, using two methods:
- 1. median Euclidean distance relative to a pool of randomly sampled 'normal' images
- 2. pairwise Euclidean distance between two images for direct comparison
Part 4:
- Example of inference of image Euclidean distance relative to an anchor (pooled 'normal' images)
- Example of two image pairwise Euclidean distance inference
'''

# PyTorch modules
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
from dataloader import MultiData
# from siamese_ROP_classes import SiameseNetwork101, img_processing, anchor_inference, twoimage_inference

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
        out = [x.detach().cpu().numpy() for x in model.forward(*images)]
        embeddings += out
        classes.append(target[0][0])
        classes.append(target[1][0])
    embeddings = np.concatenate(embeddings)

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

os.makedirs('out/'+args.prj, exist_ok=True)

'''
Part 2: Evaluation of Euclidean distance disease severity of all test set images relative to a pool of normal studies
- Take random sample of 10 "normal" images from the test set (normal image pool)
- Do pairwise comparison of each test set image with the normal image pool image (i.e. 10 euclidean distances)
- Take arithmetic median of the euclidean distances to provide a measure of disease severity
'''

if args.distance:
    # read test set path
    test = pd.read_csv('data/womac_pairs05_test.csv')
    testL = test[['ID', 'V00WOMKPL', 'painL', 'path1']].rename(
        columns={"V00WOMKPL": "womac", 'painL': "pain", 'path1': 'path'})
    testR = test[['ID', 'V00WOMKPR', 'painR', 'path2']].rename(
        columns={"V00WOMKPR": "womac", 'painR': "pain", 'path2': 'path'})
    test = pd.concat([testL, testR]).reset_index()
    test_0 = test[test['womac'] == 0]
    test_anchor = test_0.sample(n=10, random_state=42)
    test_0 = test_0.drop(test_anchor.index)
    test_other = test[test['womac'] != 0]
    print('womac0:',len(test_0))
    print('others:',len(test_other))
    test = pd.concat([test_0.sample(n=100, random_state=42), test_other.sample(n=400, random_state=42)]) #random get 700 data
    test_img = test['path'].tolist()
    labels = test['womac'].tolist()
    test_img = [root + i for i in test_img]
    img_anchor = test_anchor['path'].tolist()
    img_anchor = [root + i for i in img_anchor]

    euclidean_distance_record = []
    grade_record = [] #ground truth womac score
    image_path_record = []
    for j in range(len(test_img)):
        test = [test_img[j]]*10
        label = [labels[j]]*10
        test_set = MultiData(root=root, path=[test, img_anchor], labels=label, opt=args, mode='test', filenames=False)
        save_euclidean_distance = []
        for i in range(10):
            target_img = torch.unsqueeze(test_set.__getitem__(i)[0][0], 0).cuda()
            ref_img = torch.unsqueeze(test_set.__getitem__(i)[0][1], 0).cuda()
            output1, output2 = [x.detach().cpu() for x in net.forward(target_img, ref_img)]
            euclidean_distance = F.pairwise_distance(output1, output2).item()
            save_euclidean_distance.append(euclidean_distance)
        euclidean_distance_record.append(statistics.median(save_euclidean_distance))
        grade_record.append(labels[j])
        image_path_record.append(test_img[j].split('/')[-1])
    print('distance calculation completed')

    pooled_normal_test = pd.DataFrame({'image_path': image_path_record,
                                       'euclidean_distance': euclidean_distance_record,
                                       'grade_record': grade_record,
                                       })
    pooled_normal_test.to_csv('out/' + args.prj + '/ep' + args.epoch + '_pooled_normal_test.csv')

### visualization ###
if args.plot:
    pooled_normal_test = pd.read_csv('out/' + args.prj + '/ep' + args.epoch +'_pooled_normal_test.csv')
    # zero = pooled_normal_test[pooled_normal_test['grade_record'] == 0]
    # notzero = pooled_normal_test[pooled_normal_test['grade_record'] != 0]
    # pooled_normal_test = pd.concat([zero.sample(100), notzero.sample(700)])
    # print(len(pooled_normal_test))

    pooled_normal_test['grade_record'] = pooled_normal_test['grade_record'].astype(int)
    pooled_normal_test = pooled_normal_test[pooled_normal_test['euclidean_distance'] <= 1.25] #rule out outliers
    # pooled_normal_test['log10womac'] = np.log10(pooled_normal_test['grade_record']+1)

    # boxplot
    plt.figure()
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    sns.set(style="white")
    ax = sns.boxplot(x="grade_record", y="euclidean_distance", data=pooled_normal_test, color='white', showfliers=False)

    for i, box in enumerate(ax.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')
        # iterate over whiskers and median lines
        for j in range(5 * i, 5 * (i + 1)):
            ax.lines[j].set_color('black')

    ax = sns.swarmplot(x="grade_record", y="euclidean_distance", data=pooled_normal_test, color="grey", size=2)
    # ax = sns.scatterplot(data=pooled_normal_test,x="log10womac", y="euclidean_distance", color="grey", size=3)
    plt.xlabel('Womac', fontsize=15)
    plt.ylabel('Median Euclidean Distance', fontsize=15)
    plt.savefig('out/' + args.prj + '/ep' + args.epoch + "_EuclideanDist_boxplot_pooled_analysis_median_log.png")
    # plt.savefig('out/' + args.prj + "/EuclideanDist_scatter_pooled_analysis_median_log.png")
    plt.close()

# Spearman rank correlation
# pooled_normal_test.loc[pooled_normal_test['grade_record'] == 'No', 'grade_record'] = 0
# pooled_normal_test.loc[pooled_normal_test['grade_record'] == 'Pre-Plus', 'grade_record'] = 1
# pooled_normal_test.loc[pooled_normal_test['grade_record'] == 'Plus', 'grade_record'] = 2
#
# pooled_normal_test_[['euclidean_distance', 'grade_record']].corr(method="spearman")

if args.tsne:
    # read test set path
    test = pd.read_csv('data/womac_pairs05_test.csv')
    # test = pd.read_csv('data/womac_pairs05_train.csv')
    test = test[:100]
    test_img1 = test['path1'].tolist()
    test_img2 = test['path2'].tolist()
    labels = list(zip(test.painL, test.painR))
    test_set = MultiData(root=root, path=[test_img1, test_img2], labels=labels, opt=args, mode='test', filenames=False)
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
    color_dict = {'low':'red', 'medium':'green', 'high':'blue'}
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=color_dict[y[i]], alpha=0.5)
    leg = plt.legend(color_dict.keys())
    for i, j in enumerate(leg.legendHandles):
        j.set_color(list(color_dict.values())[i])
    plt.xticks([])
    plt.yticks([])
    plt.savefig('out/' + args.prj + '/ep' + args.epoch + "_tsne.png")
    plt.close()

# CUDA_VISIBLE_DEVICES=0 python eval.py --prj 0117_cat_lre5 --epoch 17  --tsne --distance --plot