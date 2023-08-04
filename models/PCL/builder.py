import torch
import torch.nn as nn
from random import sample
from torchvision.models import resnet

def ResNet18(low_dim=128, dataset_name='wisdm'):

    if dataset_name == 'DuckDuckGeese':
        in_channels = 1345
    elif dataset_name == 'FingerMovements':
        in_channels = 28
    elif dataset_name == 'PenDigits':
        in_channels = 2
    elif dataset_name == 'PhonemeSpectra':
        in_channels = 11
    elif dataset_name == 'StandWalkJump':
        in_channels = 4
    elif dataset_name == 'InsectWingbeat':
        in_channels = 200
    elif dataset_name == 'EigenWorms':
        in_channels = 6
    elif dataset_name == 'HAR':
        in_channels = 9
    elif dataset_name == 'SHAR':
        in_channels = 3
    elif dataset_name == 'wisdm':
        in_channels = 3
    elif dataset_name == 'epilepsy':
        in_channels = 1

   
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

class MoCo(nn.Module):

    def __init__(self, base_encoder, dim=128, r=4, m=0.999, T=0.1, mlp=False, dataset_name='wisdm'):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 4)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

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

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        print(f'batch_size:{batch_size}')#384
        print(f'self.r:{self.r}')#896
        print(f'keys.T.shape:{keys.T.shape}')#[128, 384]


        assert self.r % batch_size == 0  # for simplicity
        print('&*&*&*&*&*&*1')
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        print('&*&*&*&*&*&*2')
        ptr = (ptr + batch_size) % self.r  # move pointer
        print('&*&*&*&*&*&*3')
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

    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        if is_eval:
            # self.encoder_k.conv1 = nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1, bias=False)

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
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        # prototypical contrast
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id] print('e')
                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())]
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist()) 
                neg_proto_id = sample(neg_proto_id,self.r) #sample r negative prototypes
                neg_prototypes = prototypes[neg_proto_id]
                proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
                # compute prototypical logits
                logits_proto = torch.mm(q,proto_selected.t())
                
                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]
                logits_proto /= temp_proto
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return logits, labels, proto_logits, proto_labels
        else:
            return logits, labels, None, None


# utils
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
