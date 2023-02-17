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

def read_paired_path(csv_path):
    path_list = []
    df = pd.read_csv(csv_path)
    df_paths = df[df.columns[df.columns.str.startswith('path')]]
    for i in range(len(df_paths.columns)):
        col = df_paths[df_paths.columns[i]]
        path_list.append(col.values.tolist())
    print('reading csv at {} with {} imgs in a pair'.format(csv_path, len(df_paths.columns)))
    # labels = df[['label', 'painL', 'painR']]
    # labels = list(zip(df.label, df.painL, df.painR))
    labels = list(zip(df.label, df.V00WOMKPL.astype('int32'), df.V00WOMKPR.astype('int32'))) #pytorch metric learning
    return path_list, labels

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

class MultiData(data.Dataset):
    """
    Multiple unpaired data combined
    """
    def __init__(self, root, path, opt, mode, labels=None, transforms=None, filenames=False, index=None):
        super(MultiData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        self.subset = []

        if self.opt.load3d:
            print('load3D...')
            self.subset.append(PairedData3D(root=root, paths=path, opt=opt, mode=mode, labels=labels,
                                            transforms=transforms, filenames=filenames, index=index))
        else:
            self.subset.append(PairedData(root=root, path=paired_path[p],opt=opt, mode=mode, labels=labels,
                                          transforms=transforms, filenames=filenames, index=index))

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
    def __init__(self, root, paths, opt, mode, transforms=None, labels=None, filenames=True, index=None):
        super(PairedData3D, self).__init__()

        self.index = index
        self.opt = opt
        self.mode = mode
        self.twocrop = opt.twocrop
        self.filenames = filenames
        self.labels = labels
        self.all_paths = []
        # list shape = (imgs in a pair,subject number,img num in a knee=23)
        for pair1 in paths:
            temp = []
            for one_path in pair1:
                temp.append(glob.glob(os.path.join(root, one_path) + '*'))
            self.all_paths.append(temp)

        if self.opt.resize == 0:
            self.resize = np.array(Image.open(self.all_paths[0][0][0])).shape[1]
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
        if self.twocrop:
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
            return len(self.all_paths[0])

    def __getitem__(self, idx):
        if self.index is not None:
            index = self.index[idx]
        else:
            index = idx

        # add all the slices into the dict
        length_of_each_path = []
        filenames = []
        for i in self.all_paths:  # loop over all subjects in a pair
            filenames = filenames + i[index]
            length_of_each_path.append(len(i[index]))
        if self.twocrop:
            length_of_each_path = length_of_each_path*2

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
        labels = self.labels[index]
        if self.twocrop:
            outputs = [total[0], total[2], total[1], total[3]] # 0-2 1-3 is augmented pairs
            labels = (self.labels[index][1], self.labels[index][1], self.labels[index][2], self.labels[index][2])

        # return only images or with filenames
        if self.filenames:
            return outputs, labels, filenames
        else:
            return outputs, labels

if __name__ == '__main__':
    import argparse

    # Arguments
    parser = argparse.ArgumentParser()  # add_help=False)
    parser.add_argument('--resize', type=int, default=0, help='size for resizing before cropping, 0 for no resizing')
    parser.add_argument('--cropsize', type=int, default=256, help='size for cropping, 0 for no crop')
    parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=True, help='do 3D')
    parser.add_argument('--trd', type=float, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=True)
    parser.add_argument('--twocrop', action='store_true', dest='twocrop', default=True)
    args = parser.parse_args()

    csv_path = 'data/part_train.csv'
    root = '/media/ExtHDD02/OAIDataBase/'
    paths, labels = read_paired_path(csv_path)

    train_set = MultiData(root=root, path=paths, labels=labels, opt=args, mode='train', filenames=True, transforms=None)

    print(len(train_set.__getitem__(1)[0])) #img
    print(train_set.__getitem__(1)[1]) #label
    print(train_set.__getitem__(1)[0][0].shape)

    # for i in range(len(train_set.__getitem__(1)[0])):
    #     imgs = train_set.__getitem__(1)[0][i]
    #     imgs = torch.permute(imgs, (0, 3, 1, 2))[0]
    #     x = imgs.numpy()
    #     x = x - x.min()
    #     x = x / x.max()
        # tiff.imwrite('out/imgs/'+'cropped_'+str(i)+'.tif', x)
    # torch.Size([3, 256, 256, 23])