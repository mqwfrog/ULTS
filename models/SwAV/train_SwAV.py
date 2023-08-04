import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC

from utils import *
import resnet50 as resnet_models
from ..data_loader.dataset_loader import data_generator 


class train_SwAV(): 
    
    def __init__(self, parser):
        self.args = parser.parse_args()
        self.method = 'SwAV'
        self.dataset_name = self.args.dataset_name
        self.temperature = self.args.tau
        self.epsilon = self.args.epsilon
        self.sinkhorn_iterations = self.args.sinkhorn_iterations
        self.feat_dim = self.args.feature_size
        self.nmb_prototypes = self.args.nmb_prototypes
        self.queue_length = self.args.queue_length
        self.epoch_queue_starts = self.args.epoch_queue_starts
        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.base_lr = self.args.lr
        self.final_lr = self.args.final_lr
        self.freeze_prototypes_niters = self.args.freeze_prototypes_niters
        self.wd = self.args.decay_lr
        self.warmup_epochs = self.args.warmup_epochs
        self.start_warmup = self.args.start_warmup
        self.dist_url = self.args.dist_url
        self.world_size = self.args.world_size
        self.rank = self.args.rank
        self.local_rank = self.args.local_rank
        self.arch = self.args.arch
        self.hidden_mlp = self.args.hidden_mlp
        self.workers = self.args.workers
        self.checkpoint_freq = self.args.checkpoint_freq
        self.use_fp16 = self.args.use_fp16
        self.sync_bn = self.args.sync_bn
        self.dump_path = self.args.dump_path
        self.seed = self.args.seed

    def excute(self):
        logger = getLogger()
        fix_random_seeds(self.seed)

        if not os.path.exists(f'./checkpoints/ck_{self.dataset_name}'): 
            os.makedirs(f'./checkpoints/ck_{self.dataset_name}')

        logger, training_stats = initialize_exp(self, "epoch", "loss")

        os.environ["CUDA_VISIBLE_DEVICES"] = str('0')

     
        if self.dataset_name == 'HAR':
            from config_files.HAR_Configs import Config as Configs
        elif self.dataset_name == 'wisdm':
            from config_files.wisdm_Configs import Config as Configs
        elif self.dataset_name == 'epilepsy':
            from config_files.epilepsy_Configs import Config as Configs
        elif self.dataset_name == 'SHAR':
            from config_files.SHAR_Configs import Config as Configs
        elif self.dataset_name == 'DuckDuckGeese':
            from config_files.DuckDuckGeese_Configs import Config as Configs
        elif self.dataset_name == 'FingerMovements':
            from config_files.FingerMovements_Configs import Config as Configs
        elif self.dataset_name == 'PenDigits':
            from config_files.PenDigits_Configs import Config as Configs
        elif self.dataset_name == 'PhonemeSpectra':
            from config_files.PhonemeSpectra_Configs import Config as Configs
        elif self.dataset_name == 'StandWalkJump':
            from config_files.StandWalkJump_Configs import Config as Configs
        elif self.dataset_name == 'InsectWingbeat':
            from config_files.InsectWingbeat_Configs import Config as Configs
        elif self.dataset_name == 'EigenWorms':
            from config_files.EigenWorms_Configs import Config as Configs

        full_data_path = os.path.join('./data/', self.dataset_name)
        print(f'full_data_path:{full_data_path}')
        configs = Configs()
       
        train_dl, test_dl = data_generator(full_data_path, configs, 'self_supervised')  # train_linear
        print(f"len_trainloader：{len(train_dl)}")
        # print(f"len_validloader：{len(valid_dl)}")
        print(f"len_testloader：{len(test_dl)}")
       
        logger.info("Building data done.")

        # build model
        model = resnet_models.__dict__[self.arch](
            normalize=True,
            hidden_mlp=self.hidden_mlp,
            output_dim=self.feat_dim,
            nmb_prototypes=self.nmb_prototypes,
            dataset_name=self.dataset_name, 
        )

       
        model = model.cuda()
        if self.rank == 0:
            logger.info(model)
        logger.info("Building model done.")

        # build optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.base_lr,
            momentum=0.9,
            weight_decay=self.wd,
        )
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
       
        warmup_lr_schedule = np.linspace(self.start_warmup, self.base_lr, len(train_dl) * self.warmup_epochs)
        iters = np.arange(len(train_dl) * (self.epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array([self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (1 + \
                             math.cos(math.pi * t / (len(train_dl) * (self.epochs - self.warmup_epochs)))) for t in iters])
        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
      
        logger.info("Building optimizer done.")

        # init mixed precision
        if self.use_fp16:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
            logger.info("Initializing mixed precision done.")


        # optionally resume from a checkpoint
        to_restore = {"epoch": 0}
        restart_from_checkpoint(
            os.path.join(self.dump_path, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
            amp=apex.amp,
        )
        start_epoch = to_restore["epoch"]

        # build the queue
        queue = None
        queue_path = os.path.join(self.dump_path, "queue" + str(self.rank) + ".pth")
        if os.path.isfile(queue_path):
            queue = torch.load(queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        self.queue_length -= self.queue_length % (self.batch_size * self.world_size)

        cudnn.benchmark = True

        for epoch in range(start_epoch, self.epochs):

            # train the network for one epoch
            logger.info("============ Starting epoch %i ... ============" % epoch)

        
            # optionally starts a queue
            if self.queue_length > 0 and epoch >= self.epoch_queue_starts and queue is None:
                queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length // self.world_size,
                    self.feat_dim,
                ).cuda()
          
            scores, queue = train(train_dl, model, optimizer, epoch, lr_schedule, queue) 
            training_stats.update(scores)
           
            # save checkpoints
            if self.rank == 0:
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if self.use_fp16:
                    save_dict["amp"] = apex.amp.state_dict()
                torch.save(
                    save_dict,
                    os.path.join(self.dump_path, "checkpoint.pth.tar"),
                )
                if epoch % self.checkpoint_freq == 0 or epoch == self.epochs - 1:
                    shutil.copyfile(
                        os.path.join(self.dump_path, "checkpoint.pth.tar"),
                        os.path.join(self.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                    )
            if queue is not None:
                torch.save({"queue": queue}, queue_path)


    def train(self, train_loader, model, optimizer, epoch, lr_schedule, queue):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        softmax = nn.Softmax(dim=1).cuda()
        model.train()
        use_the_queue = False

        end = time.time()
        for it, inputs in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs[0] = inputs[0].unsqueeze(3)  
            inputs[2] = inputs[2].unsqueeze(3)  
            inputs[3] = inputs[3].unsqueeze(3)  

            print(inputs[0].shape) 
            # update learning rate
            iteration = epoch * len(train_loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]

            # normalize the prototypes
            with torch.no_grad():
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.prototypes.weight.copy_(w)

            # ============ multi-res forward passes ... ============
            embedding, output = model([inputs[0], inputs[2], inputs[3]]) 
            embedding = embedding.detach()
            bs = inputs[0].size(0)

            print(f"embedding.shape:{embedding.shape}")
            print(f"output.shape:{output.shape}")
            print(f"bs:{bs}")
            # ============ swav loss ... ============
            loss = 0
            for i, crop_id in enumerate(self.crops_for_assign): 
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)]

                    # time to use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.prototypes.weight.t()
                            ), out))
                        # fill the queue
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                    # get assignments
                    q = torch.exp(out / self.epsilon).t()
                    q = distributed_sinkhorn(q, self.sinkhorn_iterations)[-bs:] 

                subloss = 0
                print(f'bs * crop_id: bs * (crop_id + 1):{bs * crop_id}: {bs * (crop_id + 1)}')
                p = softmax(output[bs * crop_id: bs * (crop_id + 1)] / self.temperature)
                # print("dimension", p, q.size(), p.size, torch.log(p).size())
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))

                loss += subloss / (np.sum(self.nmb_crops) - 1)
            loss /= len(self.crops_for_assign)

            # ============ backward and optim step ... ============
            optimizer.zero_grad()
            if self.use_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # cancel some gradients
            if iteration < arselfs.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            optimizer.step()

            # ============ misc ... ============
            losses.update(loss.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if self.rank ==0 and it % 50 == 0:
                logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        it,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=optimizer.optim.param_groups[0]["lr"],
                    )
                )
        return (epoch, losses.avg), queue


    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            # dist.all_reduce(sum_Q)
            Q /= sum_Q

            u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.world_size * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            # dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                # dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
