import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import copy
import builtins
import math
import os
import random
import shutil
import sys
import warnings
import time
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.spatial as ss

from collections import defaultdict
from sklearn import metrics
from pynndescent import NNDescent
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader  
from ..data_loader.dataset_loader import data_generator 
import framework


class train_MHCCL():
	def __init__(self,parser):
	    	self.args = parser.parse_args()
	        self.method = 'MHCCL'
	        self.dataset_name = self.args.dataset_name
	        self.master_port = self.args.master_port
		self.workers = self.args.workers
		self.epochs = self.args.epochs
		self.start_epoch = self.args.start_epoch
		self.batch_size = self.args.batch_size
		self.lr = self.args.lr
		self.schedule = self.args.schedule
		self.momentum = self.args.momentum
		self.wd = self.args.decay_lr
		self.print_freq = self.args.print_freq
		self.resume = self.args.load_model
		self.world_size = self.args.world_size
		self.rank = self.args.rank
		self.dist_url = self.args.dist_url
		self.dist_backend = self.args.dist_backend
		self.seed = self.args.seed
		self.gpu = self.args.device_id
		self.multiprocessing_distributed = self.args.multiprocessing_distributed
		self.low_dim = self.args.feature_size
		self.posi = self.args.posi
		self.negi = self.args.negi
		self.posp = self.args.posp
		self.negp = self.args.negp
		self.moco_m = self.args.moco_m
		self.tempi = self.args.tempi
		self.tempp = self.args.tempp
		self.mlp = self.args.mlp
		self.cos = self.args.cos
		self.warmup_epoch = self.args.warmup_epoch
		self.layers = self.args.layers
		self.req_clust = self.args.req_clust
		self.protoNCE_only = self.args.protoNCE_only
		self.mask_layer0 = self.args.mask_layer0
		self.mask_others = self.args.mask_others
		self.replace_centroids = self.args.replace_centroids
		self.usetemp = self.args.usetemp
		self.mask_mode = self.args.mask_mode
		self.dist_threshold = self.args.dist_threshold
		self.proportion = self.args.proportion

	def excute(self):
		os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
		os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	    exp_dir = f'experiment_{self.dataset_name}'

	    if self.replace_centroids is True:
	        exp_dir = f'experiment_{self.dataset_name}_replace_centroids'
	    else:
	        if self.mask_layer0 is True and self.mask_others is False:
	            if self.mask_mode == 'mask_threshold':
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layer0_{self.dist_threshold}'
	            elif self.mask_mode == 'mask_proportion':
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layer0_{self.proportion}'
	            else:
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layer0'
	        elif self.mask_layer0 is False and self.mask_others is True:
	            if self.mask_mode == 'mask_threshold':
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layerothers_{self.dist_threshold}'
	            elif self.mask_mode == 'mask_proportion':
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layerothers_{self.proportion}'
	            else:
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layerothers'
	        elif self.mask_layer0 is True and self.mask_others is True:
	            if self.mask_mode == 'mask_threshold':
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layerall_{self.dist_threshold}'
	            elif self.mask_mode == 'mask_proportion':
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layerall_{self.proportion}'
	            else:
	                exp_dir = f'experiment_{self.dataset_name}_{self.mask_mode}_layerall'

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
	    

	    if not os.path.exists(exp_dir):
	        os.mkdir(exp_dir)
	        print(f'make experiment directory: {exp_dir} ')

	    ngpus_per_node = torch.cuda.device_count()
	    if self.multiprocessing_distributed:
	        # Since we have ngpus_per_node processes per node, the total world_size
	        # needs to be adjusted accordingly
	        self.world_size = ngpus_per_node * self.world_size
	        # Use torch.multiprocessing.spawn to launch distributed processes: the
	        # main_worker process function
	        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, exp_dir))
	    else:
	        # Simply call main_worker function
	        main_worker(self.gpu, ngpus_per_node, args, exp_dir)


	def main_worker(gpu, ngpus_per_node, args, exp_dir):
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
	    print('creating model')
	    model = framework.MHCCL(self.low_dim, self.posi, self.negi, self.posp, self.negp, self.moco_m,
	                            self.tempi, self.tempp, self.usetemp, self.mlp, self.dataset_name)

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
	    criterion = nn.BCEWithLogitsLoss(reduce=None).cuda(self.gpu)

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

	    if self.dataset_name == 'HAR':
	        from config_files.HAR_Configs import Config as Configs
	        configs = Configs()
	        train_loader, eval_loader = data_generator('data/HAR', configs, 'self_supervised')
	    elif self.dataset_name == 'wisdm':
	        from config_files.wisdm_Configs import Config as Configs
	        configs = Configs()
	        train_loader, eval_loader = data_generator('data/wisdm', configs, 'self_supervised')
	    elif self.dataset_name == 'epilepsy':
	        from config_files.epilepsy_Configs import Config as Configs
	        configs = Configs()
	        train_loader, eval_loader = data_generator('data/epilepsy', configs, 'self_supervised')
	    elif self.dataset_name == 'SHAR':
	        from config_files.SHAR_Configs import Config as Configs
	        configs = Configs()
	        train_loader, eval_loader = data_generator('data/SHAR', configs, 'self_supervised')
	    elif self.dataset_name == 'PenDigits':
	        from config_files.PenDigits_Configs import Config as Configs
	        configs = Configs()
	        train_loader, eval_loader = data_generator('data/PenDigits', configs, 'self_supervised')
	    elif self.dataset_name == 'EigenWorms':
	        from config_files.EigenWorms_Configs import Config as Configs
	        configs = Configs()
	        train_loader, eval_loader = data_generator('data/EigenWorms', configs, 'self_supervised')
	    elif self.dataset_name == 'FingerMovements':
	        from config_files.FingerMovements_Configs import Config as Configs
	        configs = Configs()
	        train_loader, eval_loader = data_generator('data/FingerMovements', configs, 'self_supervised')


	    for epoch in range(self.start_epoch, self.epochs):
	        cluster_result = None
	        if epoch >= self.warmup_epoch:
	            features = compute_features(train_loader, model, args)
	            print(f'features.shape:{features.shape}')
	            # placeholder for clustering result
	            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}

	            if self.gpu == 0:
	                features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
	                features = features.numpy()
	                start = time.time()

	                """ perform hierarchical clustering """
	                c, num_clust, partition_clustering, lowest_level_centroids, req_c, cluster_result = hierarchical_clustering(features, args, initial_rank=None, distance='euclidean',
	                                            ensure_early_exit=True, verbose=True, ann_threshold=40000)
	                # c: save the cluster_labels of each instance in all partitions
	                # partition_clustering: save the hierarchical merging relation of the tree structure
	                if (epoch + 1) % 10 == 0:
	                    print('Writing back the results on the provided path ...')
	                    np.savetxt(os.path.join(exp_dir, f'c_{epoch}.csv'), c, delimiter=',', fmt='%d')
	                    np.savetxt(os.path.join(exp_dir, f'num_clust_{epoch}.csv'), np.array(num_clust), delimiter=',', fmt='%d')
	                    np.savetxt(os.path.join(exp_dir, f'partition_clustering_{epoch}.csv'), np.array(partition_clustering), delimiter=',', fmt='%s')
	                    np.savetxt(os.path.join(exp_dir, f'lowest_level_centroids_{epoch}.csv'), lowest_level_centroids, delimiter=',', fmt='%s')
	                    if req_c is not None:
	                        np.savetxt(os.path.join(exp_dir, f'req_c_{epoch}.csv'), req_c, delimiter=',', fmt='%d')
	                print('Time Elapsed: {:2.2f} seconds'.format(time.time() - start))

	                # save the clustering result
	                torch.save(cluster_result, os.path.join(exp_dir, 'clusters_%d'%epoch))

	            dist.barrier()
	            # broadcast clustering result
	            for k, data_list in cluster_result.items():
	                for data_tensor in data_list:
	                    dist.broadcast(data_tensor, 0, async_op=False)
	        adjust_learning_rate(optimizer, epoch, args)

	        # train for one epoch
	        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result, c)

	        if (epoch + 1) % 10 == 0 and (not self.multiprocessing_distributed or (self.multiprocessing_distributed
	                and self.rank % ngpus_per_node == 0)):
	            save_checkpoint({
	                'epoch': epoch + 1,
	                'state_dict': model.state_dict(),
	                'optimizer' : optimizer.state_dict(),
	            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(exp_dir, epoch))


	def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None, c=None):
	    batch_time = AverageMeter('Time', ':6.3f')
	    data_time = AverageMeter('Data', ':6.3f')
	    losses = AverageMeter('Loss', ':.4e')
	    progress = ProgressMeter(
	        len(train_loader),
	        [batch_time, data_time, losses],
	        prefix="Epoch: [{}]".format(epoch))
	    # switch to train mode
	    model.train()
	    end = time.time()

	    for i, (data, target, aug1, aug2, index) in enumerate(train_loader):
	        data_time.update(time.time() - end)
	        if self.gpu is not None:
	            aug1 = aug1.unsqueeze(3)
	            aug2 = aug2.unsqueeze(3)
	            aug1, aug2 = aug1.float().cuda(self.gpu, non_blocking=True), aug2.float().cuda(self.gpu, non_blocking=True)

	        output, target, output_proto, target_proto = model(im_q=aug1, im_k=aug2, cluster_result=cluster_result, c=c, index=index)

	        # instance-wise contrastive loss (infoNCE)
	        loss = criterion(output.cuda(self.gpu, non_blocking=True), torch.tensor(target).float().cuda(self.gpu, non_blocking=True))

	        print(f'instance-wise contrastive loss:{loss}')
	        if self.protoNCE_only is True:
	            loss = loss - loss

	        # cluster-wise contrastive loss (protoNCE)
	        if output_proto is not None:
	            loss_proto = 0
	            for proto_out, proto_target in zip(output_proto, target_proto):
	                loss_proto += criterion(proto_out.cuda(self.gpu, non_blocking=True),
	                                        torch.tensor(proto_target).float().cuda(self.gpu, non_blocking=True))

	            # average loss across all partitions of prototypes
	            loss_proto /= self.layers
	            loss += loss_proto

	        losses.update(loss.item(), aug1.size(0))

	        # compute gradient and do SGD step
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()

	        # measure elapsed time
	        batch_time.update(time.time() - end)
	        end = time.time()

	        if i % self.print_freq == 0:
	            progress.display(i)

	            
	def compute_features(data_loader, model, args):
	    print('Computing features...')
	    model.eval()
	    features = torch.zeros(3 * len(data_loader.dataset), self.low_dim).cuda()

	    for i, (data, target, aug1, aug2, index) in enumerate(tqdm(data_loader)):
	        with torch.no_grad():
	            data = data.unsqueeze(3)
	            aug1 = aug1.unsqueeze(3)
	            aug2 = aug2.unsqueeze(3)

	            data, target = data.float().cuda(non_blocking=True), target.long().cuda(non_blocking=True)
	            aug1, aug2 = aug1.float().cuda(non_blocking=True), aug2.float().cuda(non_blocking=True)
	            feat = model(data, is_eval=True)
	            features[index] = feat

	            feat_aug1 = model(aug1, is_eval=True)
	            features[index + len(data_loader.dataset)] = feat_aug1
	            feat_aug2 = model(aug2, is_eval=True)
	            features[index + 2 * len(data_loader.dataset)] = feat_aug2

	    dist.barrier()
	    dist.all_reduce(features, op=dist.ReduceOp.SUM)

	    return features.cpu()


	def cool_mean(data, partition, max_dis_list=None):
	    s = data.shape[0]
	    un, nf = np.unique(partition, return_counts=True)

	    row = np.arange(0, s)
	    col = partition
	    d = np.ones(s, dtype='float32')

	    if max_dis_list is not None:
	        for i in max_dis_list:
	            data[i] = 0
	        nf = nf - 1

	    umat = sp.csr_matrix((d, (row, col)), shape=(s, len(un)))
	    cluster_rep = umat.T @ data
	    cluster_mean_rep = cluster_rep / nf[..., np.newaxis]

	    return cluster_mean_rep


	def hierarchical_clustering(x, args, initial_rank=None, distance='cosine', ensure_early_exit=True, verbose=True,  ann_threshold=40000):
	    """
	    x: input matrix with features in rows.(n_samples, n_features)
	    initial_rank: Nx1 first integer neighbor indices (optional). (n_samples, 1)
	    req_clust: set output number of clusters (optional). 
	    distance: one of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] 
	    ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
	    verbose: print verbose output.
	    ann_threshold: int (default 40000) Data size threshold below which nearest neighbors are approximated with ANNs.
	    """
	    print('Performing finch clustering')
	    req_clust = self.req_clust
	    results = {'im2cluster': [], 'centroids': [], 'density': []}

	    x = x.astype(np.float32)
	    min_sim = None

	    # calculate pairwise similarity orig_dis to find the nearest neighbor and obtain the adj matrix
	    adj, orig_dist, first_neighbors, _ = clust_rank(
	        x,
	        initial_rank,
	        distance,
	        verbose=verbose,
	        ann_threshold=ann_threshold
	    )

	    initial_rank = None

	    # eg: 520---119---31---6
	    # obtain clusters by connecting nodes using the adj matrix obtained by cluster_rank
	    u, num_clust = get_clust(adj, [], min_sim)

	    # group: the parent classes of all subclass nodes, cluster labels, num_cluster: components
	    c, mat = get_merge([], u, x) # obtain the centroids according to the partition and raw data

	    """find the points farthest from the centroids in each cluster and mask these points in next round of clustering"""
	    # orig_dist:  distance between the original samples
	    # recalculate distance between points and centroids
	    # x:(2617, 128)  mat:(521,128)  group:(2617,)

	    # step1: define cluster dict, key: cluster_label，value: id list of the cluster_label
	    # outliers_dis dic, key: cluster_label，value: distance between each point and the centroid of cluster it belongs to
	    cluster = defaultdict(list)
	    outliers_dist = defaultdict(list)

	    for i in range(0, len(u)):  # u: current partition, c: all partitions
	        cluster[u[i]].append(i)
	        outliers_dist[u[i]].append(i)

	    # step 2: compute euclidean(x[cluster[i]],mat[i]) -> find max centroids_dist

	    # max_dist_list is used to access the points farthest from centroids in each cluster,
	    # these points will be masked, and then the centroids will be recalculated
	    max_dis_list = []
	    min_dis_dict = dict()

	    """mask strategy"""
	    # mode 1: mask one point farthest from the centroid of each cluster
	    if self.mask_mode == 'mask_farthest':
	        for i in range(0, num_clust):  # calculate the distance between points and the centroids within each cluster
	            maxd = 0
	            mind = sys.maxsize
	            for j in range(0, len(cluster[i])):
	                d = ss.distance.euclidean(mat[i], x[cluster[i][j]])
	                if mind >= d:
	                    mind = d
	                    minindex = cluster[i][j]
	                if maxd <= d:
	                    maxd = d
	                    maxindex = cluster[i][j]
	            max_dis_list.append(maxindex)
	            min_dis_dict[i] = minindex

	    # mode 2: mask the points whose distance from the centroid is above the specified threshold in each cluster
	    elif self.mask_mode == 'mask_threshold':
	        for i in range(0, num_clust):
	            mind = sys.maxsize
	            for j in range(0, len(cluster[i])):
	                d = ss.distance.euclidean(mat[i], x[cluster[i][j]])
	                if mind >= d:
	                    mind = d
	                    minindex = cluster[i][j]
	                if d > self.dist_threshold:
	                    max_dis_list.append(cluster[i][j])
	            min_dis_dict[i] = minindex

	    # mode 3: mask the points with the specified proportion farthest from the centroid of each cluster
	    elif self.mask_mode == 'mask_proportion':
	        for i in range(0, num_clust):
	            mind = sys.maxsize
	            for j in range(0, len(cluster[i])):
	                d = ss.distance.euclidean(mat[i], x[cluster[i][j]])
	                if mind >= d:
	                    mind = d
	                    minindex = cluster[i][j]
	                outliers_dist[i][j] = d  # save the distance between jth point and the centroid in ith cluster
	            t = copy.deepcopy(outliers_dist[i])
	            for _ in range(round(len(outliers_dist[i]) * self.proportion)):
	                dist = max(t)
	                index = t.index(dist)
	                t[index] = 0
	                max_dis_list.append(cluster[i][index])
	            t = []
	            min_dis_dict[i] = minindex

	    # step 3: obtain the centroids according to the partition and raw data
	    """ Recalculate the centroids at layer 0 (only mask points at the first step of clustering)"""
	    if self.mask_layer0 is True:
	        mat = cool_mean(x, u, max_dis_list)

	    # clustering at layer 0 (bottom layer) end

	    # begin clustering at following layers through the while loop

	    """ replace computed prototypes with raw data """
	    if self.replace_centroids is True:
	        for i in range(0, num_clust):
	            mat[i] = x[min_dis_dict[i]]

	    lowest_level_centroids = mat

	    ''' save centroids of the bottom layer (layer 0)'''
	    lowest_centroids = torch.Tensor(lowest_level_centroids).cuda()
	    results['centroids'].append(lowest_centroids)

	    if verbose:
	        print('Level/Partition 0: {} clusters'.format(num_clust))

	    if ensure_early_exit:
	        if orig_dist.shape[-1] > 2:
	            min_sim = np.max(orig_dist * adj.toarray())


	    exit_clust = 2
	    c_ = c  # transfer value first and then mask

	    k = 1
	    num_clust = [num_clust] #int->list
	    partition_clustering = []
	    while exit_clust > 1:
	        adj, orig_dist, first_neighbors, knn_index = clust_rank(
	            mat,
	            initial_rank,
	            distance,
	            verbose=verbose,
	            ann_threshold=ann_threshold
	        )

	        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)  #u = group

	        partition_clustering.append(u)  # all partitions (u: current partition)

	        c_, mat = get_merge(c_, u, x)
	        c = np.column_stack((c, c_))

	        num_clust.append(num_clust_curr)
	        exit_clust = num_clust[-2] - num_clust_curr

	        if num_clust_curr == 1 or exit_clust <= 1:
	            num_clust = num_clust[:-1]
	            c = c[:, :-1]
	            break

	        if verbose:
	            print('Level/Partition {}: {} clusters'.format(k, num_clust[k]))

	        ''' save the controids of the bottom self.layers '''
	        # max_dis_dict = dict()
	        max_dis_list = []
	        min_dis_dict = dict()

	        """mask strategy"""
	        # mode 1: mask one point farthest from the centroid of each cluster
	        if self.mask_mode == 'mask_farthest':
	            for i in range(0, mat.shape[0]):
	                maxd = 0
	                mind = sys.maxsize
	                for j in range(0, len(cluster[i])):
	                    d = ss.distance.euclidean(mat[i], x[cluster[i][j]])
	                    if mind >= d:
	                        mind = d
	                        minindex = cluster[i][j]
	                    if maxd <= d:
	                        maxd = d
	                        maxindex = cluster[i][j]
	                max_dis_list.append(maxindex)
	                min_dis_dict[i] = minindex

	        # mode 2: mask the points whose distance from the centroid is above the specified threshold in each cluster
	        elif self.mask_mode == 'mask_threshold':
	            for i in range(0, mat.shape[0]):
	                mind = sys.maxsize
	                for j in range(0, len(cluster[i])):
	                    d = ss.distance.euclidean(mat[i], x[cluster[i][j]])
	                    if mind >= d:
	                        mind = d
	                        minindex = cluster[i][j]
	                    if d > self.dist_threshold:
	                        max_dis_list.append(cluster[i][j])
	                min_dis_dict[i] = minindex

	        # mode 3: mask the points with the specified proportion farthest from the centroid of each cluster
	        elif self.mask_mode == 'mask_proportion':
	            for i in range(0, mat.shape[0]):
	                mind = sys.maxsize
	                for j in range(0, len(cluster[i])):
	                    d = ss.distance.euclidean(mat[i], x[cluster[i][j]])
	                    if mind >= d:
	                        mind = d
	                        minindex = cluster[i][j]
	                    outliers_dist[i][j] = d
	                t = copy.deepcopy(outliers_dist[i])
	                for _ in range(round(len(outliers_dist[i]) * self.proportion)):
	                    dist = max(t)
	                    index = t.index(dist)
	                    t[index] = 0
	                    max_dis_list.append(cluster[i][index])
	                t = []
	                min_dis_dict[i] = minindex

	        """ replace computed prototypes with raw data """
	        if self.replace_centroids is True:
	            for i in range(0, mat.shape[0]):
	                mat[i] = x[min_dis_dict[i]]

	        """ Recalculate the centroids at top layers (except layer 0)"""
	        if self.mask_others is True:
	            np.savetxt('mat_beforemask.txt', mat, delimiter=',', fmt='%s')
	            mat = cool_mean(x, c_, max_dis_list)
	            np.savetxt('mat_aftermask.txt', mat, delimiter=',', fmt='%s')

	        ''' save the controids at self.layers '''
	        # if self.layers=3 means: save 533 131 32  from [533, 131, 32, 7, 2]
	        if k < self.layers:
	            centroids = torch.Tensor(mat).cuda()
	            results['centroids'].append(centroids)

	        k += 1

	    if req_clust is not None:
	        print(f'req_clust:{req_clust}')
	        print('yes')
	        if req_clust not in num_clust:
	            print('notinyes')
	            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
	            req_c = req_numclust(c[:, ind[-1]], x, req_clust, distance)
	        else:
	            print('inyes')
	            req_c = c[:, num_clust.index(req_clust)]
	    else:
	        req_c = None

	    """ save multiple partitions """
	    # save 131 32 7 from [533, 131, 32, 7, 2]
	    for i in range(0, self.layers):
	        im2cluster = [int(n[i]) for n in c]
	        im2cluster = torch.LongTensor(im2cluster).cuda()
	        results['im2cluster'].append(im2cluster)

	    return c, num_clust, partition_clustering, lowest_level_centroids, req_c, results
	    # c: NxP matrix. cluster label for every partition P. array(n_samples, n_partitions)
	    # num_clust: number of clusters. array(n_partitions)
	    # partition_clustering: list of arrays with labels indicating the centroids cluster participation per level. list of arrays of shapes equal to the values of num_clust
	    # lowest_level_centroids: feature coordinates of the lowest level centroids. array(num_clust[0], n_features)
	    # req_c: labels of required clusters (Nx1). only set if req_clust is not None.

	    
	def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	    torch.save(state, filename)
	    if is_best:
	        shutil.copyfile(filename, 'model_best.pth.tar')

	
	def adjust_learning_rate(optimizer, epoch, args):
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

	def clust_rank(
	        mat,
	        initial_rank=None,
	        metric='cosine',
	        verbose=False,
	        ann_threshold=40000):
	    knn_index = None
	    s = mat.shape[0]
	    if initial_rank is not None:
	        orig_dist = []
	    elif s <= ann_threshold:
	        # If the sample size is smaller than threshold, use metric to calculate similarity.
	        # If the sample size is larger than threshold, use PyNNDecent to speed up the calculation of nearest neighbor
	        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=metric)
	        np.fill_diagonal(orig_dist, 1e12)
	        initial_rank = np.argmin(orig_dist, axis=1)
	    else:
	        if verbose:
	            print('Using PyNNDescent to compute 1st-neighbours at this step ...')
	        knn_index = NNDescent(
	            mat,
	            n_neighbors=2,
	            metric=metric,
	            verbose=verbose)
	        result, orig_dist = knn_index.neighbor_graph
	        initial_rank = result[:, 1]
	        orig_dist[:, 0] = 1e12
	        if verbose:
	            print('Step PyNNDescent done ...')

	    sparce_adjacency_matrix = sp.csr_matrix(
	        (np.ones_like(initial_rank, dtype=np.float32),
	         (np.arange(0, s), initial_rank)),
	        shape=(s, s))  # join adjacency matrix based on Initial rank

	    return sparce_adjacency_matrix, orig_dist, initial_rank, knn_index


	def get_clust(a, orig_dist, min_sim=None):
	    # connect nodes based on adj, orig_dist, min_sim
	    # build the graph and obtain multiple components/clusters
	    if min_sim is not None:
	        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

	    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)

	    return u, num_clust


	def get_merge(partition, group, data):
	    # get_merge([], group, x)
	    # u/group: (n,)  data/x: (n, dim)
	    if len(partition) != 0:
	        _, ig = np.unique(partition, return_inverse=True)
	        partition = group[ig]
	    else:
	        partition = group

	    mat = cool_mean(data, partition, max_dis_list=None) # mat: computed centroids(k,dim)
	    # data: (n, dim)   partition: (n,)  return:(k, dim)
	    return partition, mat


	def update_adj(adj, d):
	    # Update adj, keep one merge at a time
	    idx = adj.nonzero()
	    v = np.argsort(d[idx])
	    v = v[:2]
	    x = [idx[0][v[0]], idx[0][v[1]]]
	    y = [idx[1][v[0]], idx[1][v[1]]]
	    a = sp.lil_matrix(adj.get_shape())
	    a[x, y] = 1
	    return a


	def req_numclust(c, data, req_clust, distance):
	    print('update when req_clust is specified')
	    iter_ = len(np.unique(c)) - req_clust
	    c_, mat = get_merge([], c, data)
	    for i in range(iter_):
	        adj, orig_dist, _, _ = clust_rank(mat, initial_rank=None, metric=distance)
	        adj = update_adj(adj, orig_dist)
	        u, _ = get_clust(adj, [], min_sim=None)
	        c_, mat = get_merge(c_, u, data)
	    return c_


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


