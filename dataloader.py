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

def get_transforms(crop_size, resize, additional_targets, need=('train', 'test')):
    transformations = {}
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.Resize(resize, resize),
            #A.augmentations.geometric.rotate.Rotate(limit=45, p=0.5),
            A.RandomCrop(height=crop_size, width=crop_size, p=1.),
            #A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=400),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    if 'test' in need:
        transformations['test'] = A.Compose([
            A.Resize(resize, resize),
            A.CenterCrop(height=crop_size, width=crop_size, p=1.),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    return transformations

def read_paired_path(csv_path):
    df = pd.read_csv(csv_path)
    img1_path = df['path1']
    img2_path = df['path2']
    label = df['label']
    return img1_path, img2_path, label

class MultiData(data.Dataset):
    """
    Multiple unpaired data combined
    """
    def __init__(self, root, path, opt, mode, labels=None, transforms=None, filenames=False, index=None):
        super(MultiData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        paired_path = path
        self.subset = []

        if self.opt.load3d:
            self.subset.append(PairedData3D(root=root, path1=paired_path[0], path2=paired_path[1],
                                            opt=opt, mode=mode, labels=labels, transforms=transforms, filenames=filenames, index=index))
        else:
            self.subset.append(PairedData(root=root, path=paired_path[p],
                                          opt=opt, mode=mode, labels=labels, transforms=transforms, filenames=filenames, index=index))

    def shuffle_images(self):
        for set in self.subset:
            random.shuffle(set.images)

    def __len__(self):
        return min([len(x) for x in self.subset])

    def __getitem__(self, index):
        outputs_all = []
        filenames_all = []
        if self.filenames:
            for i in range(len(self.subset)):
                outputs, labels, filenames = self.subset[i].__getitem__(index)
                outputs_all = outputs_all + outputs
                filenames_all = filenames_all + filenames
            # return {'img': outputs_all, 'labels': labels, 'filenames': filenames_all}
            return outputs_all, labels, filenames_all
        else:
            for i in range(len(self.subset)):
                outputs, labels = self.subset[i].__getitem__(index)
                outputs_all = outputs_all + outputs
            # return {'img': outputs_all, 'labels': labels}
            return outputs_all, labels


class PairedData3D(data.Dataset): # path = list of img path from csv
    def __init__(self, root, path1, path2, opt, mode, transforms=None, labels=None, filenames=False, index=None):
        super(PairedData3D, self).__init__()

        self.index = index
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        self.labels = labels

        self.pair1 = dict()
        self.pair1['path'] = path1
        self.pair1['all_path'] = [glob.glob(os.path.join(root, x) + '*') for x in path1]
        self.pair1['images'] = sorted([x.split('/')[-1] for x in list(itertools.chain(*self.pair1['all_path']))])
        self.pair2 = dict()
        self.pair2['path'] = path2
        self.pair2['all_path'] = [glob.glob(os.path.join(root, x) + '*') for x in path2]
        self.pair2['images'] = sorted([x.split('/')[-1] for x in list(itertools.chain(*self.pair2['all_path']))])

        if self.opt.resize == 0:
            self.resize = np.array(Image.open(self.pair1['all_path'][0][0])).shape[1]
        else:
            self.resize = self.opt.resize
        if self.opt.cropsize == 0:
            self.cropsize = self.resize
        else:
            self.cropsize = self.opt.cropsize

        if transforms is None:
            additional_targets = dict()
            for i in range(1, 9999):#len(self.all_path)):
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
            return len(self.images)

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        # add all the slices into the dict
        length_of_each_path = []
        filenames = []
        for i in [self.pair1, self.pair2]:  # loop over all subjects in a pair
            filenames = filenames + i['all_path'][index]
            length_of_each_path.append(len(i['all_path'][index]))
        inputs = self.load_to_dict(filenames)

        # Do augmentation
        outputs = self.get_augumentation(inputs)

        # split to different paths
        total = []
        for split in length_of_each_path:
            temp = []
            for i in range(split):
                temp.append(outputs.pop(0).unsqueeze(3))
            total.append(torch.cat(temp, 3))
        outputs = total

        # return only images or with filenames
        if self.filenames:
            return outputs, self.labels[index], filenames
        else:
            return outputs, self.labels[index]

if __name__ == '__main__':
    import argparse

    # Arguments
    parser = argparse.ArgumentParser()  # add_help=False)
    parser.add_argument('--resize', type=int, default=0, help='size for resizing before cropping, 0 for no resizing')
    parser.add_argument('--cropsize', type=int, default=256, help='size for cropping, 0 for no crop')
    parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=True, help='do 3D')
    parser.add_argument('--trd', type=float, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=False)
    args = parser.parse_args()

    csv_path = 'LRpair_path.csv'
    root = '/media/ExtHDD02/OAIDataBase/OAI_pain/full/'
    img1_paths, img2_paths, labels = read_paired_path(csv_path)

    train_set = MultiData(root=root, path=[img1_paths.tolist(), img2_paths.tolist()], labels=labels.tolist(),
                        opt=args, mode='train', filenames=True)

    print(len(train_set.__getitem__(1)))
    # print(train_set.__getitem__(1)[1])