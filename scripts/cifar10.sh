# Train cifar10
# Just replace resnet34 with other model names such as resnet50 to train on them, refer to models/__init__.py
# you can also tweak hyper-parameters, refer to utils/argparser.py

# Normal training on CIFAR10

# aug_method can be any of the following:
# [vanilla, cutmix, corr, cnc]

# train resnet
python train.py \
--dataset cifar10 \
--model resnet34 \
--wd 0.0005 \
--schedule-steps 100 150 \
--epochs 200 \
--aug cnc \
--exp_name ecml_run

# train densenet
python train.py \
--dataset cifar10 \
--model densenet100 \
--wd 0.0001 \
--schedule-steps 150 225 \
--epochs 300 \
--aug cnc \
--exp_name ecml_run

# train wide_resnet
python train.py \
--dataset cifar100 \
--model wideresnet40_2 \
--wd 0.0005 \
--nesterov 1 \
--scheduler cosine \
--epochs 100 \
--lr 0.1 \
--aug cnc \
--exp_name ecml_run