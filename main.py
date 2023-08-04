import argparse
import torch

from models.SimCLR.train_SimCLR import *
from models.BYOL.train_BYOL import *
from models.CPC.train_CPC import *
from models.SwAV.train_SwAV import *
from models.PCL.train_PCL import *
from models.MHCCL.train_MHCCL import *
from models.TS2Vec.train_TS2Vec import *
from models.TSTCC.train_TSTCC import *
from models.TLoss.train_TLoss import *
from models.TST.train_TST import *
from models.train_DeepCluster import *
from models.train_IDFD import *


parser = argparse.ArgumentParser()

# models
parser.add_argument('--uid', type=str, default='SimCLR',
                    help='Staging identifier (default: SimCLR)')

# datasets
parser.add_argument('--dataset_name', default='wisdm', type=str,
                    help='Choice of Datasets: wisdm, epilepsy, HAR, SHAR, etc.')
parser.add_argument('--feature_size', type=int, default=128,
                    help='Feature output size (default: 128')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input training batch-size')

# hyperparameters
parser.add_argument('--accumulation_steps', type=int, default=4, metavar='N',
                    help='Gradient accumulation steps (default: 4')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay_lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')
parser.add_argument('--tau', default=0.5, type=float,
                    help='Tau temperature smoothing (default 0.5)')

# gpu
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables cuda (default: False')
parser.add_argument('--device_id', type=int, default=0,
                    help='GPU device id (default: 0')

# directions
parser.add_argument('--data_dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--exp_dir', default='exp1', type=str,
                    help='Logs and checkpoints of experiments')

# modes
parser.add_argument('--training_mode', default='self_supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--load_model', type=str, default=None,
                    help='Load model to resume training for (default None)')

# SwAV
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--nmb_prototypes", default=1000, type=int,
                    help="number of prototypes") 
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")
parser.add_argument("--final_lr", type=float, default=0.0006, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=900, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs to only train with InfoNCE loss")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=64, type=int,
                    help="hidden layer dimension in projection head") #2048->64
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically") 
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")

# PCL
parser.add_argument('--master_port', type=str, default='29501',
                    help='avoid address already in use')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--pcl_r', default=4, type=int,
                    help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--num_cluster', default='6,12,24', type=str,
                    help='number of clusters')


# MHCCL
parser.add_argument('--posi', default=3, type=int,
                    help='number of positive instance pairs (default: 3)')
parser.add_argument('--negi', default=4, type=int,
                    help='number of negative instance pairs(default: 4)')
parser.add_argument('--posp', default=3, type=int,
                    help='number of positive prototype pairs (default: 3)')
parser.add_argument('--negp', default=4, type=int,
                    help='number of negative prototype pairs(default: 4)')
parser.add_argument('--tempi', default=0.2, type=float,
                    help='softmax temperature for instances')
parser.add_argument('--tempp', default=0.3, type=float,
                    help='softmax temperature for prototypes')
parser.add_argument('--layers', default=3, type=int,
                    help='save the results of bottom # layers (default 3 for wisdm)')
parser.add_argument('--req_clust', default=None, type=int,
                    help='specify the number of clusters ')
parser.add_argument('--protoNCE_only', action="store_true",
                    help='use protoNCE loss only ')
parser.add_argument('--mask_layer0', action="store_true",
                    help='mask points and recompute centroids at the bottom layer 0')
parser.add_argument('--mask_others', action="store_true",
                    help='mask points and recompute centroids at all top layers')
parser.add_argument('--replace_centroids', action="store_true",
                    help='replace computed prototypes with raw data')
parser.add_argument('--usetemp', action="store_true",
                    help='adopt temperature in loss')
parser.add_argument('--mask_mode', default='mask_farthest', type=str, choices=['mask_farthest', 'mask_threshold', 'mask_proportion'],
                    help='select the mask mode (default: mask_farthest, other values:'
                         'mask_threshold(if use, specify the dist_threshold), '
                         'mask_proportion(if use, specify the proportion')
parser.add_argument('--dist_threshold', default=0.3, type=float,
                    help='specify the distance threshold beyond which points will be masked '
                         'when select the mask_threshold mode')
parser.add_argument('--proportion', default=0.5, type=float,
                    help='specify the proportion of how much points far from the centroids will be masked '
                         'when select the mask_proportion mode')


# TS2Vec
parser.add_argument('--archive', type=str, required=True, help='The archive name that the dataset belongs to. This can be set to UCR, UEA, forecast_csv, or forecast_csv_univar')
parser.add_argument('--max_train_length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
parser.add_argument('--save_every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
parser.add_argument('--max_threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')


# TLoss
parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of hyperparameters to use ' +
                             'for training; must be a JSON file')
parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')

if __name__ == '__main__':
    args=parser.parse_args()

    if args.uid == 'SimCLR':
        train_SimCLR(parser).excute()

    elif args.uid == 'BYOL':
        train_BYOL(parser).excute()

    elif args.uid == 'CPC':
        train_CPC(parser).excute()

    elif args.uid == 'SwAV':
        train_SwAV(parser).excute()

    elif args.uid == 'PCL':
        train_PCL(parser).excute()

    elif args.uid == 'MHCCL':
        train_MHCCL(parser).excute()

    elif args.uid == 'TS2Vec':
        train_TS2Vec(parser).excute()

    elif args.uid == 'TSTCC':
        train_TSTCC(parser).excute()

    elif args.uid == 'TLoss':
        train_TLoss(parser).excute()

    elif args.uid == 'TST':
        train_TST(parser).excute()

    elif args.uid == 'DeepCluster':
        train_DeepCluster(parser).excute()

    elif args.uid == 'IDFD':
        train_IDFD(parser).excute()




