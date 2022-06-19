import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training for calibration')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=100, type=int,
                        help='seed to use')
    # Optimization options
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--total_iters', default=20000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_iter', default=0, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--log_interval', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--save_interval', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--aug', default="vanilla", type=str,
                        help='augmentation to train with')
    parser.add_argument('--nesterov', default=0, type=int)

    parser.add_argument('--in_data', default="imagenet", type=str,
                        help='augmentation to train with')
    parser.add_argument('--out_data', default="inaturalist", type=str,
                        help='augmentation to train with')
    parser.add_argument('--ood_method', default="msp", type=str,
                        help='augmentation to train with')

    parser.add_argument('--exp_name', default="", type=str,
                        help='experiment name to distinguish b/w different experiments')
    parser.add_argument('--scheduler', default="sgd", type=str,
                        help='experiment name to distinguish b/w different experiments')
                        
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--train-batch-size', default=64, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch-size', default=100, type=int, metavar='N',
                        help='test batchsize')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--schedule-steps', type=int, nargs='+', default=[],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1, help='LR is multiplied by this on schedule.')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--model', default='resnet20', type=str, metavar='MNAME')
    parser.add_argument('--optimizer', default='sgd', type=str, metavar='ONAME')
    
    return parser.parse_args()