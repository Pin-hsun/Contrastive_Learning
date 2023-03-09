import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
import random
import itertools
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tifffile as tiff

def read_paired_path(csv_path, n=None):
    # path_list = [n*[2]]
    path_list = []
    df = pd.read_csv(csv_path)
    if n is None:
        n = len(df)
    df_paths = df[df.columns[df.columns.str.startswith('path')]][:n]
    for index, row in df_paths.iterrows():
        path_list.append(row.values)
    print('reading csv at {} with {} imgs in a pair'.format(csv_path, len(df_paths.columns)))
    labels = list(zip(df.painL[:n], df.painR[:n]))
    # labels = list(zip(df.label, df.V00WOMKPL.astype('int32'), df.V00WOMKPR.astype('int32'))) #pytorch metric learning
    return path_list, labels

def get_transforms(crop_size, resize, additional_targets, need=('train', 'test')):
    transformations = {}
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.Resize(resize, resize),
            A.augmentations.geometric.rotate.Rotate(limit=45, p=0.5),
            A.RandomCrop(height=crop_size, width=crop_size, p=1.),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=400),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    if 'test' in need:
        transformations['test'] = A.Compose([
            A.Resize(resize, resize),
            A.CenterCrop(height=crop_size, width=crop_size, p=1.),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=400),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    return transformations


class PairedData3D(data.Dataset): # path = list of pairs (img paths) from csv
    def __init__(self, root, paths, opt, mode, transforms=None, labels=None, filenames=True, index=None):
        super(PairedData3D, self).__init__()

        self.index = index
        self.all_paths = []
        for pair in paths:
            temp = []
            for path in pair:
                temp.append(glob.glob(os.path.join(root, path) + '*'))
            self.all_paths.append(temp)
        self.opt = opt
        self.mode = mode
        self.twocrop = opt.twocrop
        self.filenames = filenames
        self.labels = labels
        if self.opt.resize == 0:
            self.resize = np.array(Image.open(glob.glob(root+paths[0][0]+'*')[0])).shape[1]
        else:
            self.resize = self.opt.resize
        if self.opt.cropsize == 0:
            self.cropsize = self.resize
        else:
            self.cropsize = self.opt.cropsize

        if transforms is None:
            additional_targets = dict()
            for i in range(1, 23):
                additional_targets[str(i).zfill(4)] = 'image'
            self.transforms = get_transforms(crop_size=self.cropsize,
                                             resize=self.resize,
                                             additional_targets=additional_targets)[mode]
        else:
            self.transforms = transforms


    def load_img(self, path):
        x = Image.open(path)
        x = np.array(x).astype(np.float32)

        if self.opt.trd > 0:
            x[x >= self.opt.trd] = self.opt.trd

        if x.max() > 0:  # scale to 0-1
            x = x / x.max()

        if len(x.shape) == 2:  # if grayscale
            x = np.expand_dims(x, 2)
        if not self.opt.gray:
            if x.shape[2] == 1:
                x = np.concatenate([x]*3, 2)
        return x

    def load_to_dict(self, names):
        out = dict()
        for i in range(len(names)):
            out[str(i).zfill(4)] = self.load_img(names[i])
        out['image'] = out.pop('0000')  # the first image in albumentation need to be named "image"
        return out

    def get_augumentation(self, inputs):
        outputs = []
        if self.twocrop & (self.mode=='train'):
            augmented = [self.transforms(**inputs), self.transforms(**inputs)]
            augmented[0]['0000'] = augmented[0].pop('image')
            augmented[1]['0000'] = augmented[1].pop('image')
            for i in range(2):
                for k in sorted(list(augmented[i].keys())):
                    if self.opt.n01:
                        outputs = outputs + [augmented[i][k], ]
                    else:
                        if augmented[i][k].shape[0] == 3:
                            outputs = outputs + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(augmented[i][k]), ]
                        elif augmented[i][k].shape[0] == 1:
                            outputs = outputs + [transforms.Normalize(0.5, 0.5)(augmented[i][k]), ]
        else:
            augmented = self.transforms(**inputs)
            augmented['0000'] = augmented.pop('image')  # 'change image back to 0'
            for k in sorted(list(augmented.keys())):
                if self.opt.n01:
                    outputs = outputs + [augmented[k], ]
                else:
                    if augmented[k].shape[0] == 3:
                        outputs = outputs + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(augmented[k]), ]
                    elif augmented[k].shape[0] == 1:
                        outputs = outputs + [transforms.Normalize(0.5, 0.5)(augmented[k]), ]
        return outputs

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.all_paths)

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        # load imgs to dict and do augmentation
        augmented_img_pairs = []
        id = []
        for subject in self.all_paths[idx]:
            id.append(subject[0].split('/')[-1])
            input_dic = self.load_to_dict(subject)
            outputs = self.get_augumentation(input_dic)
            augmented_img_pairs.append(outputs)
        #  augmented_img_pairs = [2[23[torch.Size([3, 256, 256])]]]

        # turn augmented_img_pairs into torch.Size([3, 256, 256, 23])
        total = []
        for subject in augmented_img_pairs:
            temp = []
            for i in range(len(subject)):
                temp.append(subject.pop(0).unsqueeze(3))
            total.append(torch.cat(temp, 3))

        outputs = torch.stack(total) #torch.Size([4, 3, 256, 256, 23])
        labels = torch.tensor((self.labels[index][0], self.labels[index][1]))
        if self.twocrop & (self.mode=='train'):
            labels = torch.cat([labels, labels], dim=0)
        # return only images or with filenames
        if self.filenames:
            return outputs, labels, tuple(id)
        else:
            return outputs, labels

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse

    # Arguments
    parser = argparse.ArgumentParser()  # add_help=False)
    parser.add_argument('--resize', type=int, default=0, help='size for resizing before cropping, 0 for no resizing')
    parser.add_argument('--cropsize', type=int, default=256, help='size for cropping, 0 for no crop')
    parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=True, help='do 3D')
    parser.add_argument('--trd', type=float, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=True)
    parser.add_argument('--twocrop', action='store_true', dest='twocrop', default=False)
    args = parser.parse_args()
    print(args)

    csv_path = 'data/SupCon_pairs03_part_train.csv'
    root = '/media/ExtHDD02/OAIDataBase/'
    paths, labels = read_paired_path(csv_path)

    train_set = PairedData3D(root=root, paths=paths, labels=labels, opt=args, mode='train', filenames=True, transforms=None)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=2, shuffle=False, pin_memory=True)
    print(train_set[1][0])  # img torch.Size([2, 3, 256, 256, 23])
    # print(train_set.__getitem__(1)[1])  # label tensor([1, 0])
    print(train_set.__getitem__(1)[2])  # id ['9000798_00_L_001.tif', '9000798_00_R_009.tif']

    # for i in range(train_set.__len__()):
        # print(len(train_set.__getitem__(i)[0])) #img
        # print(train_set.__getitem__(i)[1]) #label
        # print(train_set.__getitem__(i)[2]) #filenames
        # print(train_set.__getitem__(1)[0][0].shape)

    # for batch_idx, (data, labels, id) in enumerate(train_loader):
    #     labels = labels.view(labels.shape[0]*labels.shape[1])
    #     data = data.view(data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4], data.shape[5])
    #     print(id)
    #     print(labels)
    #     print(data.shape)

    # for i in range(len(train_set.__getitem__(1)[0])):
    #     imgs = train_set.__getitem__(1)[0][i]
    #     imgs = torch.permute(imgs, (0, 3, 1, 2))[0]
    #     x = imgs.numpy()
    #     x = x - x.min()
    #     x = x / x.max()
        # tiff.imwrite('out/imgs/'+'cropped_'+str(i)+'.tif', x)
    # torch.Size([3, 256, 256, 23])