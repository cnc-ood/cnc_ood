from .cifar10 import get_train_valid_test_loader as cifar10loader

from .cifar100 import get_train_valid_test_loader as cifar100loader

from .svhn import get_train_valid_test_loader as svhnloader

from .imagenet import get_train_valid_test_loader as imagenetloader

from .tinyimagenet import get_train_valid_test_loader as tinyimagenetloader

dataloader_dict = {
    "cifar10" : cifar10loader,
    "cifar100" : cifar100loader,
    "svhn" : svhnloader,
    "imagenet" : imagenetloader,
    "tinyimagenet" : tinyimagenetloader
}

dataset_nclasses_dict = {
    "cifar10" : 10,
    "cifar100" : 100,
    "svhn" : 10,
    "imagenet" : 1000,
    "tinyimagenet" : 200
}

dataset_classname_dict = {
    "cifar10" : ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],

    "cifar100" : ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
                'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 
                'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
                'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                'whale', 'willow_tree', 'wolf', 'woman', 'worm'],
                
    "svhn" : [f"{i}" for i in range(10)],

    "imagenet" : [f"{i}" for i in range(1000)],

    "tinyimagenet" : [f"{i}" for i in range(200)],
}
