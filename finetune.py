# Finetune CnC, cutmix or Corr augmentation networks on ImageNet-1K

import os
import torch
import torchvision

import torch.optim as optim

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path, freeze_weights
from utils import AverageMeter

from solvers.runners import train, train_aug, test

from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from time import localtime, strftime, time

import logging

def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    
    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    
    # prepare save path
    if len(args.exp_name):
        model_save_pth = f"{args.checkpoint}/{args.dataset}/{current_time}{create_save_path(args)}_{args.exp_name}"
    else:
        model_save_pth = f"{args.checkpoint}/{args.dataset}/{current_time}{create_save_path(args)}"

    checkpoint_dir_name = model_save_pth

    if not os.path.isdir(model_save_pth):
        mkdir_p(model_save_pth)

    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(model_save_pth, "train.log")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Setting up logging folder : {model_save_pth}")

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]

    # Add extra class to model
    if args.aug == "cutmix" or args.aug == "corr" or args.aug == "cnc":
        num_classes += 1
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    if args.model == "resnet34":
        model = torchvision.models.resnet34(pretrained=False)
        # Freeze the network first
        # freeze_weights(model)
        # Re-Init the last layer
        model.fc = torch.nn.Linear(512, num_classes)
    # elif args.model == "resnet50":
    #     model = torchvision.models.resnet50(pretrained=True)
    #     # Freeze the network first
    #     freeze_weights(model)
    #     # Re-Init the last layer
    #     model.fc = torch.nn.Linear(512 * 4, num_classes)
    else:
        logging.error("Not a valid model to train with")

    # send to gpu
    model = torch.nn.DataParallel(model)
    model.cuda()
    
    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, corruptloader, testloader = dataloader_dict[args.dataset](args)

    logging.info(f"Setting up optimizer : {args.optimizer}")

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), 
                              lr=args.lr, 
                              momentum=args.momentum, # ) 
                              weight_decay=args.weight_decay)
    
    criterion = torch.nn.CrossEntropyLoss()
    test_criterion = torch.nn.CrossEntropyLoss()
    
    logging.info(f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)

    start_iter = args.start_iter
    
    best_acc = 0.
    best_acc_stats = {"top1" : 0.0}

    # choose train func
    train_func = train # if args.aug == "vanilla" else train_aug

    train_meter = AverageMeter()
    top1_train_meter = AverageMeter()

    train_iter = recycle(trainloader)
    corr_iter = recycle(corruptloader)

    start = time()

    for iter in range(start_iter, args.total_iters):

        train_loss, top1_train = train_func(train_iter, corr_iter, model, optimizer, criterion)
        train_meter.update(train_loss)
        top1_train_meter.update(top1_train)

        if iter % args.log_interval == 0:
            logging.info("Iter {}/{} stats: avg_train_loss : {:.4f} | avg_top1_train : {:.4f} | lr : {:.5f}".format(
                iter,
                args.total_iters,
                train_meter.avg,
                top1_train_meter.avg,
                get_lr(optimizer)
            ))

            train_meter.reset()
            top1_train_meter.reset()
        
        scheduler.step()

        if iter % args.save_interval == 0:
            logging.info("Running validation now...")
            # save best accuracy model
            test_loss, top1, top3, top5 = test(testloader, model, test_criterion)
            time_taken = time() - start
            start = time()

            logging.info("Iter {}/{} validation stats: test_loss : {:.4f} | top1 : {:.4f} | top3 : {:.5f} | top5 : {:.5f} | time_taken : {:.5f} mins".format(
                iter,
                args.total_iters,
                test_loss,
                top1,
                top3,
                top5,
                time_taken / 60
            ))

            is_best = top1 > best_acc
            best_acc = max(best_acc, top1)

            save_checkpoint({
                    'epoch': iter,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'model' : args.model
                }, is_best, checkpoint=model_save_pth)
            
            # Update best stats
            if is_best:
                best_acc_stats = {
                    "top1" : top1,
                    "top3" : top3,
                    "top5" : top5
                }

    logging.info("training completed...")
    logging.info("The stats for best trained model on test set are as below:")
    logging.info(best_acc_stats)