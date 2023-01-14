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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# loading siamese neural network model from the main.py output
output_folder_name = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints',
                                  'epoch' + str(args.epoch) +'.pth')
root = os.environ.get('DATASET')
# output_dir = working_path + "scripts/histories/" + output_folder_name
# net = SiameseNetwork101().cuda()
# net.load_state_dict(torch.load(output_dir + "/siamese_ROP_model.pth"))
net = torch.load(output_folder_name)
net.eval()
# history = pickle.load(open(output_dir + "/history_training.pckl", "rb"))

'''
Part 2: Evaluation of Euclidean distance disease severity of all test set images relative to a pool of normal studies
- Take random sample of 10 "normal" images from the test set (normal image pool)
- Do pairwise comparison of each test set image with the normal image pool image (i.e. 10 euclidean distances)
- Take arithmetic median of the euclidean distances to provide a measure of disease severity
'''

# read test set path
paired_csv = pd.read_csv('data/womac_pairs_score.csv')
train_index, test_index = train_test_split(list(range(len(paired_csv))), test_size=0.3, random_state=42)
test = paired_csv.iloc[test_index]
test = test[(test['V00WOMKPL'] >= 1) | (test['V00WOMKPR'] >= 1)] #don't use knee without pain
test_img = test['path1'].tolist() + test['path2'].tolist()
labels = test['V00WOMKPL'].tolist() + test['V00WOMKPR'].tolist()
test_img = [root + i for i in test_img]

# anchor images from the random 10 images
random10 = pd.read_csv('data/anchor.csv')
img_anchor = random10['path'].tolist()
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
# print(image_path_record)
# print(womac_record)
# print(euclidean_distance_record)

pooled_normal_test = pd.DataFrame({'image_path': image_path_record,
                                   'euclidean_distance': euclidean_distance_record,
                                   'grade_record': grade_record,
                                   })

pooled_normal_test.to_csv('out/' + args.prj +'/pooled_normal_test.csv')

### visualization ###
# pooled_normal_test = pd.read_csv('/pooled_normal_test.csv')

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

ax = sns.swarmplot(x="grade_record", y="euclidean_distance", data=pooled_normal_test, color="grey", size=3)
plt.xlabel('Plus Disease Classification', fontsize=15)
plt.ylabel('Median Euclidean Distance', fontsize=15)
plt.savefig('out/' + args.prj + "/EuclideanDist_boxplot_pooled_analysis_median.png")
plt.close()

# Spearman rank correlation
# pooled_normal_test.loc[pooled_normal_test['grade_record'] == 'No', 'grade_record'] = 0
# pooled_normal_test.loc[pooled_normal_test['grade_record'] == 'Pre-Plus', 'grade_record'] = 1
# pooled_normal_test.loc[pooled_normal_test['grade_record'] == 'Plus', 'grade_record'] = 2
#
# pooled_normal_test_[['euclidean_distance', 'grade_record']].corr(method="spearman")