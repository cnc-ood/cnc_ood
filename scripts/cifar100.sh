# train with CnC
python train.py \
--dataset cifar100 \
--model resnet34 \
--wd 0.0005 \
--schedule-steps 100 150 \
--epochs 200 \
--aug cnc \
--exp_name ecml_run_1

# to train cifar100 on other methods, refer to scripts/cifar10.sh