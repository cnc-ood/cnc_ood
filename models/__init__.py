from .resnet import ResNet34, ResNet50
from .densenet import DenseNet100
from .wide_resnet import WideResNet40_2
from .resnet_tinyimagenet import resnet50 as rn_tin

model_dict = {
    # resnet models can be used for cifar10/100, svhn

    "resnet34" : ResNet34,
    "resnet50" : ResNet50,
    "densenet100" : DenseNet100,

    "wideresnet40_2" : WideResNet40_2,

    "resnet_tinyimagenet" : rn_tin
}