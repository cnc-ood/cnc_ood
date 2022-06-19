import os
import torch
import numpy as np

import torch.optim as optim
from solvers.loss import CrossEntropyWithSoftTargets

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path

from solvers.runners import train, train_aug, test

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from time import localtime, strftime, time

import logging

from utils.logger import Logger

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

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
    if args.aug == "cutmix" or args.aug == "corr" or args.aug == "cnc" or args.aug == "hypmix":
        num_classes += 1
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes)
    model = torch.nn.DataParallel(model)
    model.cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, corruptloader, testloader = dataloader_dict[args.dataset](args)

    logging.info(f"Setting up optimizer : {args.optimizer}")

    # if args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay, nesterov=args.nesterov)
    
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = CrossEntropyWithSoftTargets()
    test_criterion = torch.nn.CrossEntropyLoss()
    
    if "cosine" in args.scheduler:
        logging.info(f"Using consine annealing")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer,
                        lr_lambda=lambda step: cosine_annealing(
                            step,
                            args.epochs * len(trainloader),
                            1,  # since lr_lambda computes multiplicative factor
                            1e-6 / args.lr))
    else:
        logging.info(f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)

    logger = Logger(os.path.join(model_save_pth, "train_metrics.txt"))
    logger.set_names(["lr", "train_loss", "top1_train", "test_loss", "top1", "top3", "top5"])

    start_epoch = args.start_epoch
    
    best_acc = 0.
    best_acc_stats = {"top1" : 0.0}

    # choose train func
    train_func = train if args.aug == "vanilla" else train_aug

    for epoch in range(start_epoch, args.epochs):

        start_time = time()

        logging.info('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, get_lr(optimizer)))
        
        train_loss, top1_train = train_func(args, trainloader, corruptloader, model, optimizer, criterion, scheduler)
        test_loss, top1, top3, top5 = test(testloader, model, test_criterion)

        time_taken = time() - start_time

        if "sgd" in args.scheduler:
            scheduler.step()

        logging.info("End of epoch {} stats: train_loss : {:.4f} | test_loss : {:.4f} | top1_train : {:.4f} | top1 : {:.4f}".format(
            epoch+1,
            train_loss,
            test_loss,
            top1_train,
            top1
        ))

        logging.info("Time taken for epoch : {:.2f} mins".format(time_taken/60))

        logger.append([get_lr(optimizer), train_loss, top1_train, test_loss, top1, top3, top5])

        # save best accuracy model
        is_best = top1 > best_acc
        best_acc = max(best_acc, top1)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'dataset' : args.dataset,
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