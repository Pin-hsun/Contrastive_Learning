import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import copy

def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


def append_parameters(blocks):
    parameters = [list(x.parameters()) for x in blocks]
    all_parameters = []
    for pars in parameters:
        for par in pars:
            all_parameters.append(par)
    return all_parameters


def to_freeze(pars):
    for par in pars:
        par.requires_grad = False


def to_unfreeze(pars):
    for par in pars:
        par.requires_grad = True


class ResnetFeatures(nn.Module):
    def __init__(self):
        super(ResnetFeatures, self).__init__()
        self.resnet = getattr(models, resnet_name)(pretrained=pretrained)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.fmap_c = fmap_c

        to_freeze(list(self.resnet.parameters()))
        #pars = append_parameters([getattr(self.resnet, x)[-1] for x in ['layer4']])
        #to_unfreeze(pars)

        print_num_of_parameters(self.resnet)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], self.fmap_c, 8, 8)
        return x

class SiameseNetwork101(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """
    def __init__(self):
        super(SiameseNetwork101, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = models.resnet101(pretrained=True)
        self.cnn1.fc = nn.Linear(2048, 3) # mapping input image to a 3 node output
        to_freeze(list(self.cnn1.parameters()))
        pars = append_parameters([getattr(self.cnn1, x)[-1] for x in ['layer4']])
        to_unfreeze(pars)

    def forward(self, x):
        x = self.cnn1(x)
        x = x.view(x.shape[0], 3, 1, 1)
        return x

class MRPretrained(nn.Module):
    def __init__(self, args_m):
        super(MRPretrained, self).__init__()

        self.args_m = args_m

        if args_m.backbone == 'alexnet':
            self.fmap_c = 256
        elif args_m.backbone in ['densenet121']:
            self.fmap_c = 1024
        elif args_m.backbone in ['resnet50', 'resnet101']:
            self.fmap_c = 2048
        else:
            self.fmap_c = 512

        self.features = self.get_encoder(args_m)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if args_m.fuse == 'cat':
            self.fc = nn.Sequential(nn.Linear(self.fmap_c*23, self.fmap_c), nn.Linear(self.fmap_c, 5))
        if args_m.fuse == 'max':
            self.fc = nn.Sequential(nn.Linear(self.fmap_c, 5))

        self.fuse = args_m.fuse
        self.fc_use = args_m.fc_use

    def get_encoder(self, args_m):
        if args_m.backbone =='siamese_resnet101': #set fc_use == false
            features = SiameseNetwork101()
        elif args_m.backbone.startswith('resnet'):
            features = ResnetFeatures(args_m.backbone, pretrained=args_m.pretrained, fmap_c=self.fmap_c)
        elif args_m.backbone == 'SqueezeNet':
            features = getattr(models, args_m.backbone)().features
        else:
            features = getattr(models, args_m.backbone)(pretrained=args_m.pretrained).features
        return features

    def reshape(self, x):
        B = x.shape[0]
        x = x.permute(0, 4, 1, 2, 3)  # (B, 23, 3, 224, 224)
        x = x.reshape(B * x.shape[1], x.shape[2], x.shape[3], x.shape[4])  # (B*23, 3, 224, 224)
        return x, B

    def cat_features(self, x, B): # concatenate across the slices
        x = self.features(x)  # (B*23, 3, 1, 1) alex=(B*23, 256, 7, 7)
        x = self.avg(x)  # (B*23, 3, 1, 1) alex=(B*23, 256, 1, 1)
        x = x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3])  # (B, 256, 3, 1, 1)
        xcat = x.view(B, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # (B, 23*256, 1, 1)
        features = torch.squeeze(xcat, 3)
        features = torch.squeeze(features, 2) # (B, 23*256)
        if self.fc_use:
            features = self.fc(features)
        return features

    def max_features(self, x, B): # max-pooling across the slices
        x = self.features(x)  # (B*23, 512, 7, 7)
        x = self.avg(x)  # (B*23, 512, 1, 1)
        x = x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3])  # (B, 23, 512, 1, 1)
        features, _ = torch.max(x, 1)  # (B, 512, 1, 1)
        features = torch.squeeze(features, 3)
        features = torch.squeeze(features, 2) #torch.Size([1, 2048])
        if self.fc_use:
            features = self.fc(features)
        return features

    def forward(self, img_list):   #[B, 3, 256, 256, 23]
        out = []
        for i in range(len(img_list)):
            x = img_list[i]
            # dummies
            features = None  # features we want to further analysis
            # reshape
            x, B = self.reshape(x)
            # fusion
            if self.fuse == 'cat':  # concatenate across the slices
                features = self.cat_features(x, B)
            if self.fuse == 'max':  # max-pooling across the slices
                features = self.max_features(x, B)
            out.append(features)

        return out


if __name__ == '__main__':
    # net = MRPretrained('resnet101', pretrained=True, fmap_c=2048)
    import argparse

    # Arguments
    parser = argparse.ArgumentParser()  # add_help=False)
    parser.add_argument('--backbone',default='siamese_resnet101', type=str)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--fuse', type=str, default='cat', help='cat or max')
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--fc_use', action='store_true', default=False)

    args_m = parser.parse_args()
    net = MRPretrained(args_m= args_m)
    out = net.forward([torch.rand(1, 3, 256, 256, 23), torch.rand(1, 3, 256, 256, 23), torch.rand(1, 3, 256, 256, 23)])
    print(len(out))
    print(out[0].shape)