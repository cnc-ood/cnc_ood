# Mainly will be used for ImageNet-2012 as in-distribution testing

import os
import torch
from utils.test_utils import iterate_data_msp, iterate_data_cnc, get_measures, data_path_dict 
import torchvision as tv
from utils import parse_args

from models import model_dict
from datasets import dataset_nclasses_dict

import logging

def mk_id_ood(args):
    """Returns train and validation datasets."""
    crop = 224

    # imagenet normalize
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        normalize,
    ])

    in_datadir = data_path_dict[args.in_data]
    out_datadir = data_path_dict[args.out_data]

    in_set = tv.datasets.ImageFolder(in_datadir, val_tx)
    out_set = tv.datasets.ImageFolder(out_datadir, val_tx)

    logging.info(f"Using an in-distribution dataset {args.in_data} with {len(in_set)} images.")
    logging.info(f"Using an out-of-distribution dataset {args.out_data} with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader

def run_eval(model, in_loader, out_loader, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logging.info("Running test...")

    if args.ood_method in ["msp"]:
        logging.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logging.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)

    elif args.ood_method in ["cnc", "cutmix", "corr"]:
        logging.info("Processing in-distribution data...")
        in_scores = iterate_data_cnc(in_loader, model)
        logging.info("Processing out-of-distribution data...")
        out_scores = iterate_data_cnc(out_loader, model)
    # elif args.score == 'ODIN':
    #     logging.info("Processing in-distribution data...")
    #     in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    #     logger.info("Processing out-of-distribution data...")
    #     out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    logging.info('============Results for {}============'.format(args.ood_method))
    logging.info(f'in-dataset: {args.in_data} | out-dataset: {args.out_data}')
    logging.info('AUROC: {:.4f}'.format(auroc))
    logging.info('AUPR (In): {:.4f}'.format(aupr_in))
    logging.info('AUPR (Out): {:.4f}'.format(aupr_out))
    logging.info('FPR95: {:.4f}'.format(fpr95))

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    args = parse_args()

    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            # logging.FileHandler(filename=os.path.join(model_save_pth, "train.log")),
                            logging.StreamHandler()
                        ])

    in_set, out_set, in_loader, out_loader = mk_id_ood(args)
    num_classes = dataset_nclasses_dict[args.dataset]
    if args.ood_method == "cnc":
        num_classes += 1

    logging.info(f"Using model : {args.model}")
    logging.info("Loading trained model from : {}".format(os.path.dirname(args.checkpoint)))
    model = model_dict[args.model](num_classes=num_classes)
    model = torch.nn.DataParallel(model)
    
    assert os.path.isfile(args.checkpoint), "Please provide a valid model path"
    model_dict = torch.load(args.checkpoint)
    model.load_state_dict(model_dict["state_dict"])
    model.cuda()

    run_eval(model, in_loader, out_loader, args, len(in_set.classes))