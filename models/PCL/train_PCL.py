import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import builder
from ..data_loader.dataset_loader import data_generator 


class train_PCL():
    def __init__(self,parser):
        self.args = parser.parse_args()
        self.method = 'PCL'
        self.dataset_name = self.args.dataset_name
        self.master_port = self.args.master_port
        self.momentum = self.args.momentum
        self.print_freq = self.args.print_freq
        self.resume = self.args.load_model
        self.dist_backend = self.args.dist_backend
        self.temperature = self.args.tau
        self.epsilon = self.args.epsilon
        self.sinkhorn_iterations = self.args.sinkhorn_iterations
        self.low_dim = self.args.feature_size
        self.pcl_r = self.args.pcl_r
        self.moco_m = self.args.moco_m
        self.mlp = self.args.mlp
        self.cos = self.args.cos
        self.num_cluster = self.args.num_cluster
        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.base_lr = self.args.lr
        self.wd = self.args.decay_lr
        self.warmup_epoch = self.args.warmup_epochs
        self.start_epoch = self.args.start_warmup
        self.dist_url = self.args.dist_url
        self.world_size = self.args.world_size
        self.rank = self.args.rank
        self.local_rank = self.args.local_rank
        self.arch = self.args.arch
        self.workers = self.args.workers
        self.checkpoint_freq = self.args.checkpoint_freq
        self.seed = self.args.seed
        self.gpu = self.args.device_id

    def excute(self):
        # model_names = sorted(name for name in models.__dict__
        # if name.islower() and not name.startswith("__")
        # and callable(models.__dict__[name]))

        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '6'

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if self.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')

        if self.dist_url == "env://" and self.world_size == -1:
            self.world_size = int(os.environ["WORLD_SIZE"])

        self.distributed = self.world_size > 1 or self.multiprocessing_distributed
        
        self.num_cluster = self.num_cluster.split(',')
        
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        
        ngpus_per_node = torch.cuda.device_count()
        if self.multiprocessing_distributed:
            print(f'***************{self.multiprocessing_distributed}*************')
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.world_size = ngpus_per_node * self.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            print(f'main_worker:{main_worker}')
            print(f'ngpus_per_node:{ngpus_per_node}')
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            main_worker(self.gpu, ngpus_per_node, args)


    def main_worker(self, gpu, ngpus_per_node, args):
        self.gpu = gpu
        
        if self.gpu is not None:
            print("Use GPU: {} for training".format(self.gpu))

        # suppress printing if not master    
        if self.multiprocessing_distributed and self.gpu != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass
            
        if self.distributed:
            if self.dist_url == "env://" and self.rank == -1:
                self.rank = int(os.environ["RANK"])
            if self.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.rank = self.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
        # create model
        print("=> creating model '{}'".format(self.arch))
        model = pcl.builder.MoCo(
            models.__dict__[self.arch],
            self.low_dim, self.pcl_r, self.moco_m, self.temperature, self.mlp, self.dataset_name)

        if self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                model.cuda(self.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.batch_size = int(self.batch_size / ngpus_per_node)
                self.workers = int((self.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            model = model.cuda(self.gpu)
            # comment out the following line for debugging
            raise NotImplementedError("Only DistributedDataParallel is supported.")
        else:
            # AllGather implementation (batch shuffle, queue update, etc.) in
            # this code only supports DistributedDataParallel.
            raise NotImplementedError("Only DistributedDataParallel is supported.")

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(self.gpu)

        optimizer = torch.optim.SGD(model.parameters(), self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        # optionally resume from a checkpoint
        if self.resume:
            if os.path.isfile(self.resume):
                print("=> loading checkpoint '{}'".format(self.resume))
                if self.gpu is None:
                    checkpoint = torch.load(self.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.gpu)
                    checkpoint = torch.load(self.resume, map_location=loc)
                self.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))

        cudnn.benchmark = True

        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        

        print(f'self.dataset_name:{self.dataset_name}')
        if self.dataset_name == 'HAR':
            from config_files.HAR_Configs import Config as Configs
            configs = Configs()
            in_channels = 9
            train_loader, eval_loader = data_generator('data/HAR', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'wisdm':
            from config_files.wisdm_Configs import Config as Configs
            configs = Configs()
            in_channels = 3
            train_loader, eval_loader = data_generator('data/wisdm', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'epilepsy':
            from config_files.epilepsy_Configs import Config as Configs
            configs = Configs()
            in_channels = 1
            train_loader, eval_loader = data_generator('data/epilepsy', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'SHAR':
            from config_files.SHAR_Configs import Config as Configs
            configs = Configs()
            in_channels = 3
            train_loader, eval_loader = data_generator('data/SHAR', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'PenDigits':
            from config_files.PenDigits_Configs import Config as Configs
            configs = Configs()
            in_channels = 2
            train_loader, eval_loader = data_generator('data/PenDigits', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'EigenWorms':
            from config_files.EigenWorms_Configs import Config as Configs
            configs = Configs()
            in_channels = 6
            train_loader, eval_loader = data_generator('data/EigenWorms', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'FingerMovements':
            from config_files.FingerMovements_Configs import Config as Configs
            configs = Configs()
            in_channels = 28
            train_loader, eval_loader = data_generator('data/FingerMovements', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'StandWalkJump':
            from config_files.StandWalkJump_Configs import Config as Configs
            configs = Configs()
            in_channels = 4
            train_loader, eval_loader = data_generator('data/StandWalkJump', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'PhonemeSpectra':
            from config_files.PhonemeSpectra_Configs import Config as Configs
            configs = Configs()
            in_channels = 11
            train_loader, eval_loader = data_generator('data/PhonemeSpectra', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'DuckDuckGeese':
            from config_files.DuckDuckGeese_Configs import Config as Configs
            configs = Configs()
            in_channels = 1345
            train_loader, eval_loader = data_generator('data/DuckDuckGeese', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'InsectWingbeat':
            from config_files.InsectWingbeat_Configs import Config as Configs
            configs = Configs()
            in_channels = 200
            train_loader, eval_loader = data_generator('data/InsectWingbeat', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')
        elif self.dataset_name == 'CharacterTrajectories':
            from config_files.CharacterTrajectories_Configs import Config as Configs
            configs = Configs()
            in_channels = 3
            train_loader, eval_loader = data_generator('data/CharacterTrajectories', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(eval_loader):{len(eval_loader)}')

     

        for epoch in range(self.start_epoch, self.epochs):
            
            cluster_result = None
            if epoch>=self.warmup_epoch:
                # compute momentum features for center-cropped images
                # features = compute_features(eval_loader, model, args)
                features = compute_features(train_loader, model, args)#0105mqw


                # placeholder for clustering result
                cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
                for num_cluster in self.num_cluster:
                    cluster_result['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                    cluster_result['centroids'].append(torch.zeros(int(num_cluster), self.low_dim).cuda())
                    cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())

                if self.gpu == 0:
                    features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                    features = features.numpy()
                    cluster_result = run_kmeans(features, args)  #run kmeans clustering on master node
                    # save the clustering result
                    torch.save(cluster_result, os.path.join(self.exp_dir, 'clusters_%d'%epoch))
                print('*1')
                dist.barrier()
                print('*2')
                # broadcast clustering result
                for k, data_list in cluster_result.items():
                    for data_tensor in data_list:                
                        dist.broadcast(data_tensor, 0, async_op=False)
                print('*3')
            # if self.distributed:
            #     train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)
            print('*4')
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)
            print('*5')

            if (epoch+1)%5==0 and (not self.multiprocessing_distributed or (self.multiprocessing_distributed
                    and self.rank % ngpus_per_node == 0)):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(self.exp_dir,epoch))
            print('*6')

    def train(self, train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
        print('train function here!!!')
        print('#1')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        acc_inst = AverageMeter('Acc@Inst', ':6.2f')   
        acc_proto = AverageMeter('Acc@Proto', ':6.2f')
        print('#2')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, acc_inst, acc_proto],
            prefix="Epoch: [{}]".format(epoch))
        print('#3')
        # switch to train mode
        model.train()
        print('#4')
        end = time.time()

        for i, (data, target, aug1, aug2, index) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if self.gpu is not None:
                data = data.unsqueeze(3) 
                aug1 = aug1.unsqueeze(3)  
                aug2 = aug2.unsqueeze(3) 
                data, target = data.float().cuda(self.gpu, non_blocking=True), target.long().cuda(self.gpu, non_blocking=True)
                aug1, aug2 = aug1.float().cuda(self.gpu, non_blocking=True), aug2.float().cuda(self.gpu, non_blocking=True)
                # index = index.cuda(self.gpu, non_blocking=True)

            output, target, output_proto, target_proto = model(im_q=aug1, im_k=aug2, cluster_result=cluster_result, index=index)
            print('#6')
            # InfoNCE loss
            loss = criterion(output, target)
            # loss = loss - loss #mqw
            print('#7')
            # ProtoNCE loss
            if output_proto is not None:
                loss_proto = 0
                for proto_out, proto_target in zip(output_proto, target_proto):
                    loss_proto += criterion(proto_out, proto_target)  
                    accp = accuracy(proto_out, proto_target)[0] 
                    # acc_proto.update(accp[0], images[0].size(0))
                    acc_proto.update(accp[0], aug1.size(0))
                    
                # average loss across all sets of prototypes
                loss_proto /= len(self.num_cluster) 
                loss += loss_proto
            print('#8')
            # losses.update(loss.item(), images[0].size(0))
            losses.update(loss.item(), aug1.size(0))
            acc = accuracy(output, target)[0] 
            # acc_inst.update(acc[0], images[0].size(0))
            acc_inst.update(acc[0], aug1.size(0))
            print('#9')
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                progress.display(i)

                
    def compute_features(self, eval_loader, model, args):
        print('Computing features...')
        model.eval()
        features = torch.zeros(len(eval_loader.dataset),self.low_dim).cuda()

        for i, (data, target, aug1, aug2, index) in enumerate(tqdm(eval_loader)):
            with torch.no_grad():
                data = data.unsqueeze(3)  
                aug1 = aug1.unsqueeze(3)  
                aug2 = aug2.unsqueeze(3) 
                data, target = data.float().cuda(non_blocking=True), target.long().cuda(non_blocking=True)
                aug1, aug2 = aug1.float().cuda(non_blocking=True), aug2.float().cuda(non_blocking=True)
                feat = model(data, is_eval=True)
                features[index] = feat

        dist.barrier()        
        dist.all_reduce(features, op=dist.ReduceOp.SUM)     
        return features.cpu()

        
    def run_kmeans(self, x, args):
        """
        Args:
            x: data to be clustered
        """
        
        print('performing kmeans clustering')
        results = {'im2cluster':[],'centroids':[],'density':[]}
        print('1')
        for seed, num_cluster in enumerate(self.num_cluster):
            # intialize faiss clustering parameters
            d = x.shape[1]
            k = int(num_cluster)
            print(f'k:{k}')
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5 
            clus.seed = seed
            clus.max_points_per_centroid = 1000 
            clus.min_points_per_centroid = 10 
            print('2')
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = self.gpu    
            index = faiss.GpuIndexFlatL2(res, d, cfg)
            print('3')
            clus.train(x, index)   
            print('3+1')
            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
            print('3+2')
            im2cluster = [int(n[0]) for n in I]
            print('4')
            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
            print('5')
            # sample-to-centroid distances for each cluster 
            Dcluster = [[] for c in range(k)]          
            for im,i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])
            print('6')
            # concentration estimation (phi)        
            density = np.zeros(k)
            for i,dist in enumerate(Dcluster):
                if len(dist)>1:
                    d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                    density[i] = d
            print('7')
            #if cluster only has one point, use the max to estimate its concentration        
            dmax = density.max()
            for i,dist in enumerate(Dcluster):
                if len(dist)<=1:
                    density[i] = dmax
            print('8')
            density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
            density = self.temperature*density/density.mean()  #scale the mean to temperature 
            print('9')
            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids = nn.functional.normalize(centroids, p=2, dim=1)
            print('10')
            im2cluster = torch.LongTensor(im2cluster).cuda()               
            density = torch.Tensor(density).cuda()

            results['centroids'].append(centroids)
            results['density'].append(density)
            results['im2cluster'].append(im2cluster)    
            
        return results

        
    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')


    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)


    class ProgressMeter(object):
        def __init__(self, num_batches, meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix

        def display(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print('\t'.join(entries))

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'


    def adjust_learning_rate(self, optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        lr = self.lr
        if self.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / self.epochs))
        else:  # stepwise lr schedule
            for milestone in self.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

