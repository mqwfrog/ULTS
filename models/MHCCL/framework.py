import torch
import torch.nn as nn
import numpy as np
from random import sample
from torchvision.models import resnet

def ResNet18(low_dim=128, dataset_name='wisdm'):
    if dataset_name == 'HAR':
        in_channels = 9
    elif dataset_name == 'SHAR':
        in_channels = 3
    elif dataset_name == 'wisdm':
        in_channels = 3
    elif dataset_name == 'epilepsy':
        in_channels = 1
    elif dataset_name == 'FingerMovements':
        in_channels = 28
    elif dataset_name == 'PenDigits':
        in_channels = 2
    elif dataset_name == 'EigenWorms':
        in_channels = 6

    net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], low_dim)
    if dataset_name=='wisdm':
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=8, stride=1, padding=4, bias=False)
    elif dataset_name=='HAR':
        net.conv1 = nn.Conv2d(9, 64, kernel_size=8, stride=1, padding=4, bias=False)
    elif dataset_name=='epilepsy':
        net.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=1, padding=43, bias=False)
    elif dataset_name == 'SHAR':
        net.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=1, padding=4, bias=False)
    else:
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    net.maxpool = nn.Identity()
    return net

class MHCCL(nn.Module):
    def __init__(self, dim=32, posi=1, negi=1, posp=1, negp=1, m=0.999, tempi=0.1, tempp=0.3, usetemp=False, mlp=False, dataset_name='wisdm'):
        """
        dim: representation dimension
        posi: positive instances
        negi: negative instances
        posp: positive prototypes
        negp: negative prototypes
        m: momentum for updating key encoder (default: 0.999)
	tempi: instance-level temperature
	tempp: cluster-level temperature
        usetemp: whether to use softmax temperature
        mlp: whether to use mlp projection
	dataset_name: dataset name
        """
        super(MHCCL, self).__init__()

        self.negi = negi
        self.posi = posi
        self.negp = negp
        self.posp = posp
        self.m = m
        self.tempi = tempi
        self.tempp = tempp
        self.usetemp = usetemp

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = ResNet18(low_dim=dim, dataset_name=dataset_name)
        self.encoder_k = ResNet18(low_dim=dim, dataset_name=dataset_name)


        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # if use neg queue:
        # self.queue_size = 256  # queue size
        # self.register_buffer("queue", torch.randn(dim, self.queue_size))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        print('-----momentum update-----')
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.negi  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, c=None, index=None):
        """
        Input:
            im_q: a batch of query series
            im_k: a batch of key series
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """

        if is_eval:
            k = self.encoder_k(im_q)
            k = nn.functional.normalize(k, dim=1)
            return k

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        print(f'q.shape:{q.shape}')  # aug1 [128,128]
        print(f'k.shape:{k.shape}')  # aug2 [128,128]

        # if cluster_result is not None:
        proto_labels = []
        proto_logits = []

        """instance-level contrastive learning only uses the 0-th partition"""
        p0_label = {} #dict (key:index, value:label)
        label_index = {} #dict
        index_u = {}
        for u in range(0, index.shape[0]):
            index_u[index[u].item()] = u
            p0_label[index[u].item()] = cluster_result['im2cluster'][0][index[u]].item()
        # find keys(ids) with same value(cluster label) in dict p0_label
        for key, value in p0_label.items():
            label_index.setdefault(value, []).append(key)

        posid = {}
        negid = {}
        neg_instances = [[] for _ in range(len(p0_label))]
        pos_instances = [[] for _ in range(len(p0_label))]
        all_instances = [[] for _ in range(len(p0_label))]

        for i in p0_label:
            posid[i] = label_index[p0_label[i]].copy() #all candidate pos instances(if not enough, copy itself)
            if(len(posid[i])) < self.posi:
                for _ in range(0, self.posi - len(posid[i])):
                    posid[i].append(i)
            negid[i] = [x for x in index.tolist() if x not in posid[i]]
            # print(f'negid[i]:{negid[i]}')
            if (len(posid[i])) > self.posi:
                posid[i] = sample(posid[i], self.posi) #if len = self.posi, preserve
            negid[i] = sample(negid[i], self.negi)
            # have obtained posid and negid, then find the corresponding representations and concat
            # pos[dim, 2*posi]
            # neg[dim, 2*negi]
            # all=pos+neg [dim, 2*posi+2*negi]
            for m in range(len(posid[i])):
                if posid[i][m] == i:
                    pos_instances[index_u[i]].append(k[index_u[posid[i][m]]])
                    pos_instances[index_u[i]].append(k[index_u[posid[i][m]]])#all candidate pos instances(if not enough, copy itself)
                else:
                    pos_instances[index_u[i]].append(q[index_u[posid[i][m]]])
                    pos_instances[index_u[i]].append(k[index_u[posid[i][m]]])
            pos_instances[index_u[i]] = torch.stack(pos_instances[index_u[i]])

            for n in range(len(negid[i])):
                neg_instances[index_u[i]].append(q[index_u[negid[i][n]]])
                neg_instances[index_u[i]].append(k[index_u[negid[i][n]]])
            neg_instances[index_u[i]] = torch.stack(neg_instances[index_u[i]])

            all_instances[index_u[i]] = torch.cat([pos_instances[index_u[i]], neg_instances[index_u[i]]], dim=0)

        all_instances = torch.stack(all_instances) #[batch_size, 2*posi+2*negi, dim]

        # q: [n,c],  all: [n,m+r,c],  compute logits 
        # q[n,c] -> newq[n,1,c]     all[n,m+r,c] -> all[n,c,m+r]
        # newq[n,1,c] x all[n,c,m+r] = logits[n,1,m+r]

        all_instances = torch.reshape(all_instances, (all_instances.shape[0], all_instances.shape[2], all_instances.shape[1]))
        # [batch_size, 2*posi+2*negi, dim] -> [batch_size, dim, 2*posi+2*negi]

        # logits of instances
        newq = q.unsqueeze(1)
        # q[batch_size,dim] -> newq[batch_size,1,dim]

        logits = torch.einsum('nab,nbc->nac', [newq, all_instances])
        # [batch_size,1,dim] x [batch_size, dim, 2*posi+2*negi] = [batch_size,1,2*posi+2*negi]

        logits = logits.squeeze(1)

        # nc,c(m+r) ->n(m+r)
        # [batchsize,dim] * [dim,(2pos+2neg)*batchsize] = [batchsize, (2pos+2neg)*batchsize]
        # logits = torch.einsum('nc,ck->nk', [q, all_instances])

        # apply temperature
        if self.usetemp:
            print('----------usetemp-----------')
            logits /= self.tempi

        # labels of instances
        temp_label = np.zeros(self.posi*2 + self.negi*2)
        temp_label[0: self.posi*2] = 1
        labels = np.tile(temp_label, (q.shape[0], 1)) # [B,2posi+2negi] each row has 2posi label1 and 2negi label0
        # print(f'labels of instances:{labels}')
        print(f'labels of instances.shape:{labels.shape}')

        """cluster-level contrastive learning uses multiple partitions"""
        for n, (im2cluster, prototypes) in enumerate(
                zip(cluster_result['im2cluster'], cluster_result['centroids'])):
            print(f'n:{n}') # partition-layer

            # get positive prototypes
            pos_proto_id = im2cluster[index]

            pos_prototypes = prototypes[pos_proto_id]

            all_proto_id = [i for i in range(im2cluster.max() + 1)] 

            new = pos_proto_id.split(1, 0)
            neg_proto_id = []
            new_pos_proto_id = [] # the cluster label each instance belonging to

            pos_next_partition_label = {}
            neg_next_partition_label = [{} for _ in range(len(pos_proto_id))]

            # maxlen = 0
            pdict = {} #partition_dict

            for i in range(len(pos_proto_id)):
                new_pos_proto_id.append(new[i].tolist()) # pos prototype
                """sample negative prototypes - random select neg"""
                # neg_proto_id.append(sample(list(set(all_proto_id) - set(new[i].tolist())), self.negp))

                """mask fake negative prototypes"""
                neg_proto_id.append(list(set(all_proto_id) - set(new[i].tolist())))
                # neg_proto_id[i]=list(set(all_proto_id) - set(new[i].tolist()))
                m = c.T
                if new[i] not in pdict.keys():
                    pos_next_partition_label[i] = m[n + 1][np.argwhere(m[n] == int(new[i]))[0][0]]
                    pdict[int(new[i])] = pos_next_partition_label[i]

                mask_list = []
                for j in range(0, len(neg_proto_id[i])):
                    if neg_proto_id[i][j] not in pdict.keys():
                        neg_next_partition_label[i][j] = m[n + 1][np.argwhere(m[n] == int(neg_proto_id[i][j]))[0][0]]
                        pdict[int(neg_proto_id[i][j])] = neg_next_partition_label[i][j]
                        if pos_next_partition_label[i] == neg_next_partition_label[i][j]:
                            mask_list.append(neg_proto_id[i][j]) # all fake negs that need to be masked
                    else:
                        if pdict[int(new[i])] == pdict[neg_proto_id[i][j]]:
                            mask_list.append(neg_proto_id[i][j])  # all fake negs that need to be masked

                for a in range(0, len(mask_list)):
                    neg_proto_id[i].remove(mask_list[a])
                    new_pos_proto_id[i].append(mask_list[a])

                """1- random sample n negative prototypes after masking"""
                neg_proto_id[i] = sample(neg_proto_id[i], self.negp)
                # print(f'after masking_len(neg_proto_id[i]):{len(neg_proto_id[i])}')
                # if len(neg_proto_id[i]) >= maxlen:
                #     maxlen = len(neg_proto_id[i])


                # pos prototype : 1 current centroid + (pos-1) other centroids with same parent)
                if len(new_pos_proto_id[i]) <= self.posp - 1:
                    for _ in range(0, self.posp - len(new_pos_proto_id[i])):
                        new_pos_proto_id[i].append(new_pos_proto_id[i][0])
                new_pos_proto_id[i] = [new_pos_proto_id[i][0]] + sample(new_pos_proto_id[i][1:], self.posp - 1)


            neg_prototypes = torch.zeros([pos_prototypes.shape[0], self.negp, pos_prototypes.shape[1]]).cuda()
            new_pos_prototypes = torch.zeros([pos_prototypes.shape[0], self.posp, pos_prototypes.shape[1]]).cuda()

            """2- all negative prototypes after masking"""
            for i in range(len(neg_proto_id)): #pos_proto_id
                neg_prototypes[i] = prototypes[neg_proto_id[i]]

            for i in range(len(new_pos_proto_id)):
                new_pos_prototypes[i] = prototypes[new_pos_proto_id[i]]

            # if use neg queue:
            # keys = concat_all_gather(k) #note:no name k
            # batch_size = keys.shape[0]
            # ptr = int(self.queue_ptr)
            # new_neg_prototypes = self.queue[:, ptr: ptr + batch_size * self.negp]
            # proto_selected = torch.cat([new_pos_prototypes, new_neg_prototypes], dim=1)
            # self.queue = torch.cat([proto_selected.detach(), new_neg_prototypes], dim=1)[:, :self.queue_size]

            proto_selected = torch.cat([new_pos_prototypes, neg_prototypes], dim=1) #[batch_size, pos+neg, dim]

            # compute cluster-wise logits/prototypes
            # q[batch_size, dim]    proto_selected[batch_size, posp+negp, dim]   compute logits_proto 
            # q[n,c] -> newq[n,1,c]     all[n,m+r,c] -> all[n,c,m+r]

            newq = q.unsqueeze(1)
            print(f'newq.shape:{newq.shape}')

            proto_selected = torch.reshape(proto_selected, (proto_selected.shape[0], proto_selected.shape[2], proto_selected.shape[1]))
            # [batch_size, posp+negp, dim] -> [batch_size, dim, posp+negp]
            print(f'proto_selected.shape:{proto_selected.shape}')
            # newq[n,1,c] x all[n,c,m+r] = logits[n,1,m+r]

            logits_proto = torch.einsum('nab,nbc->nac', [newq, proto_selected])
            # [batch_size,1,dim] x [batch_size, dim, posp+negp] = [batch_size, 1, posp+negp]
            print(f'logits_proto.shape:{logits_proto.shape}')
            # print(f'logits_proto:{logits_proto}')

            logits_proto = logits_proto.squeeze(1)
            print(f'logits_proto.shape:{logits_proto.shape}')
            # print(f'logits_proto:{logits_proto}')

            # labels of prototypes
            temp_proto_label = np.zeros(self.posp + self.negp)
            temp_proto_label[0: self.posp] = 1
            labels_proto = np.tile(temp_proto_label, (q.shape[0], 1))  # [B,2posi+2negi] each row has posp label1 and negp label0
            print(f'labels of prototypes.shape:{labels_proto.shape}')
            # print(f'labels of prototypes:{labels_proto}')

            # scaling temperatures for the selected prototypes
            # temp_proto = torch.zeros([batch_size, (self.negp + 1)*batch_size]).cuda()  # [batch_size,(1+n)*batch_size]
            if self.usetemp:
                logits_proto /= self.tempp

            proto_labels.append(labels_proto)
            proto_logits.append(logits_proto)

        return logits, labels, proto_logits, proto_labels


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
