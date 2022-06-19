import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy
import numpy as np

def train(args, trainloader, corruptloader, model, optimizer, criterion, scheduler):
    # switch to train mode
    model.train()

    losses_in = AverageMeter()
    losses_out = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, data in bar:
        
        inputs, targets = data
        # inputs_corr, targets_corr = data_corrupt

        inputs, targets = inputs.cuda(), targets.cuda()
        # inputs_corr = inputs_corr.cuda()

        batch_size = inputs.shape[0]

        input_ = inputs

        # compute output
        output_ = model(input_)

        outputs = output_[:batch_size]
        # outputs_corr = output_[batch_size:]

        loss_in = criterion(outputs, targets)
        # targets_corr = torch.full((outputs.shape[0],), fill_value=outputs.shape[1]-1, dtype=torch.long).cuda()
        # targets_corr = torch.full((outputs.shape[0], outputs.shape[1]), fill_value=(1/outputs.shape[1]), dtype=torch.float).cuda() # create uniform targets
        # import pdb; pdb.set_trace()
        # loss_out = criterion(outputs_corr, targets_corr)
        loss = loss_in

        # measure accuracy and record loss
        prec1, = accuracy(outputs, targets, topk=(1, ))

        losses_in.update(loss_in.item(), inputs.size(0))
        # losses_out.update(loss_out.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))

        if "cosine" in args.scheduler:
            scheduler.step()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | Loss_in: {lossin:.8f} | Loss_out: {lossout:.8f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses_in.avg + losses_out.avg,
                    lossin=losses_in.avg,
                    lossout=losses_out.avg,
                    top1=top1.avg
                    ))

    return (losses_in.avg + losses_out.avg, top1.avg)

def train_aug(args, trainloader, corruptloader, model, optimizer, criterion, scheduler):
    # switch to train mode
    model.train()

    losses_in = AverageMeter()
    losses_out = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(zip(trainloader, corruptloader)), total=len(trainloader))
    for batch_idx, (data, data_corrupt) in bar:
        
        inputs, targets = data
        inputs_corr, targets_corr = data_corrupt

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs_corr = inputs_corr.cuda()

        batch_size = inputs.shape[0]

        input_ = torch.cat((inputs, inputs_corr), 0)

        # compute output
        output_ = model(input_)

        outputs = output_[:batch_size]
        outputs_corr = output_[batch_size:]

        loss_in = criterion(outputs, targets)
        targets_corr = torch.full((outputs_corr.shape[0],), fill_value=outputs_corr.shape[1]-1, dtype=torch.long).cuda()
        # targets_corr = torch.full((outputs.shape[0], outputs.shape[1]), fill_value=(1/outputs.shape[1]), dtype=torch.float).cuda() # create uniform targets
        # import pdb; pdb.set_trace()
        loss_out = criterion(outputs_corr, targets_corr)
        loss = loss_in + loss_out

        # measure accuracy and record loss
        prec1, = accuracy(outputs, targets, topk=(1, ))

        losses_in.update(loss_in.item(), inputs.size(0))
        losses_out.update(loss_out.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))

        if "cosine" in args.scheduler:
            scheduler.step()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | Loss_in: {lossin:.8f} | Loss_out: {lossout:.8f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses_in.avg + losses_out.avg,
                    lossin=losses_in.avg,
                    lossout=losses_out.avg,
                    top1=top1.avg
                    ))

    return (losses_in.avg + losses_out.avg, top1.avg)

def train_hypmix(trainloader, corruptloader, model, optimizer, criterion):
    # switch to train mode
    model.train()

    losses_in = AverageMeter()
    losses_out = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(zip(trainloader, corruptloader)), total=len(trainloader))
    for batch_idx, (data, data_corrupt) in bar:
        
        inputs, targets = data
        inputs_corr, targets_corr = data_corrupt

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs_corr = inputs_corr.cuda()

        batch_size = inputs.shape[0]

        input_ = torch.cat((inputs, inputs_corr), 0)

        # compute output
        output_ = model(input_)

        outputs = output_[:batch_size]
        outputs_corr = output_[batch_size:]

        loss_in = criterion(outputs, targets)
        targets_corr = torch.full((outputs.shape[0],), fill_value=outputs.shape[1]-1, dtype=torch.long).cuda()
        loss_out = criterion(outputs_corr, targets_corr)
        loss = loss_in + loss_out

        # measure accuracy and record loss
        prec1, = accuracy(outputs, targets, topk=(1, ))

        losses_in.update(loss_in.item(), inputs.size(0))
        losses_out.update(loss_out.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | Loss_in: {lossin:.8f} | Loss_out: {lossout:.8f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses_in.avg + losses_out.avg,
                    lossin=losses_in.avg,
                    lossout=losses_out.avg,
                    top1=top1.avg
                    ))

    return (losses_in.avg + losses_out.avg, top1.avg)

@torch.no_grad()
def test(testloader, model, criterion):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))


        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg, top3.avg, top5.avg)

@torch.no_grad()
def get_logits_from_model_dataloader(testloader, model):
    """Returns torch tensor of logits and targets on cpu"""
    # switch to evaluate mode
    model.eval()

    all_targets = None
    all_outputs = None

    bar = tqdm(testloader, total=len(testloader), desc="Evaluating logits")
    for inputs, targets in bar:
        inputs = inputs.cuda()
        # compute output
        outputs = model(inputs)
        # to numpy
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

    return torch.from_numpy(all_outputs), torch.from_numpy(all_targets)

    
