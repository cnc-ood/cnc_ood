# train resnet50 on tiny imagenet

python train.py \
--dataset tinyimagenet \
--model resnet_tinyimagenet \
--wd 0.0005 \
--train-batch-size 128 \
--schedule-steps 50 75 \
--epochs 100 \
--aug cnc \
--exp_name ecml_run