"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.
Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from data.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import torchvision.datasets
from randaugment import rand_augment_transform
# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

# Data transformation with augmentation


def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]

# Dataset


class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None, template=None, top_k=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        # select top k class
        if top_k:
            # only select top k in training, in case train/val/test not matching.
            if 'train' in txt:
                max_len = max(self.labels) + 1
                dist = [[i, 0] for i in range(max_len)]
                for i in self.labels:
                    dist[i][-1] += 1
                dist.sort(key=lambda x: x[1], reverse=True)
                # saving
                torch.save(dist, template + '_top_{}_mapping'.format(top_k))
            else:
                # loading
                dist = torch.load(template + '_top_{}_mapping'.format(top_k))
            selected_labels = {item[0]: i for i,
                               item in enumerate(dist[:top_k])}
            # replace original path and labels
            self.new_img_path = []
            self.new_labels = []
            for path, label in zip(self.img_path, self.labels):
                if label in selected_labels:
                    self.new_img_path.append(path)
                    self.new_labels.append(selected_labels[label])
            self.img_path = self.new_img_path
            self.labels = self.new_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        try:
            sample = Image.open(path).convert('RGB')
        except:
            print(f'Fail to open image {path}, try the next')
            index += 1
            path = self.img_path[index]
            label = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

# Load datasets


def load_data(data_root, dataset, phase, top_k_class=None, cifar_imb_ratio=None, s_aug=False):
    """
    Parameters:
        s_aug: whether to use RandAugment
    """
    cifar_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    cifar_transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    if dataset == 'cifar10_LT':
        print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR10(
            phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    elif dataset == 'cifar100_LT':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(
            phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    elif dataset == 'cifar10':
        print('====> Loading balanced CIFAR10...')
        set_ = torchvision.datasets.CIFAR10(train=True if phase == "train" else False, root=data_root,
                                            transform=cifar_transform_train if phase == "train" else cifar_transform_val)
    elif dataset == 'cifar100':
        print('====> Loading balanced CIFAR100...')
        set_ = torchvision.datasets.CIFAR100(train=True if phase == "train" else False, root=data_root,
                                             transform=cifar_transform_train if phase == "train" else cifar_transform_val)
    else:
        txt_split = phase
        txt = './data/%s/%s_%s.txt' % (dataset, dataset, txt_split)
        template = './data/%s/%s' % (dataset, dataset)

        print('Loading data from %s' % (txt))

        if dataset == 'iNaturalist18':
            print('===> Loading iNaturalist18 statistics')
            key = 'iNaturalist18'
        else:
            key = 'default'
        rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
        if not s_aug:
            if phase not in ['train', 'val']:
                transform = get_data_transform('test', rgb_mean, rgb_std, key)
            else:
                transform = get_data_transform(phase, rgb_mean, rgb_std, key)
        elif s_aug:
            if phase not in ['train', 'val']:
                transform = get_data_transform('test', rgb_mean, rgb_std, key)
            elif phase == 'val':
                transform = get_data_transform(phase, rgb_mean, rgb_std, key)
            else:
                normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192)) if dataset == 'iNaturalist18' \
                    else transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ra_params = dict(translate_const=int(
                    224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in (0.485, 0.456, 0.406)]), )
                augmentation = [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    rand_augment_transform(
                        'rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
                    transforms.ToTensor(),
                    normalize]
                transform = transforms.Compose(augmentation)
        print('Use data transformation:', transform)

        set_ = LT_Dataset(data_root, txt, transform,
                          template=template, top_k=top_k_class)

    print(f'len of {dataset} = ', len(set_))
    return set_

    # if train_sampler and phase == 'train':
    #     print('=====> Using sampler: ', train_sampler)
    #     # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
    #     # print('=====> Sampler parameters: ', sampler_dic['params'])
    #     return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
    #                     sampler=train_sampler,
    #                     #    sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
    #                     num_workers=num_workers)
    # else:
    #     print('=====> No sampler.')
    #     print('=====> Shuffle is %s.' % (shuffle))
    #     return DataLoader(dataset=set_, batch_size=batch_size,
    #                       shuffle=shuffle, num_workers=num_workers)
