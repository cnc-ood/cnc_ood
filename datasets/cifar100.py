import torch

from torchvision import datasets
from torchvision import transforms
from torch.utils import data

import logging

from augmentations import aug_dict

transform_pre = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

transform_post = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# train set = 45000
# val set = 5000
# test set = 10000
def get_train_valid_test_loader(args):
    train_set = datasets.CIFAR100(root='../scratch/data', train=True, download=True, transform=transform_train)

    # Define augmentations
    if args.aug == "cutmix":
        train_set_corr = datasets.CIFAR100(root='../scratch/data', train=True, download=True, transform=transform_train)
    else:
        train_set_corr = datasets.CIFAR100(root='../scratch/data', train=True, download=True, transform=transform_pre)

    logging.info("workers being used : {}".format(args.workers))

    train_loader = data.DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.workers, shuffle=True)
    corr_loader = data.DataLoader(train_set_corr, batch_size=args.train_batch_size, num_workers=args.workers, shuffle=True, collate_fn=aug_dict[args.aug])

    test_set = datasets.CIFAR100(root='../scratch/data', train=False, download=True, transform=transform_test)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch_size, num_workers=args.workers, drop_last=False)

    return train_loader, corr_loader, test_loader

def get_datasets(args):
    trainset = datasets.CIFAR100(root='../scratch/data', train=True, download=True, transform=None)
    testset = datasets.CIFAR100(root='../scratch/data', train=False, download=True, transform=None)
    return trainset, testset

def get_transforms():
    return transform_train, transform_test