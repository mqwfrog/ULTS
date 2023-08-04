import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time

from models import *
from utils import *
from data import *
from loss import *

from torch.utils.data.dataloader import DataLoader  
from .data_loader.dataset_loader import data_generator 

class train_SimCLR():

    def __init__(self,parser):
        self.args=parser.parse_args()
        self.train_loader=None
        self.test_loader=None
        self.logger=None
        # Setup tensorboard
        self.use_tb = self.args.exp_dir is not None
        self.log_dir = self.args.exp_dir
        # Set cuda
        self.use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.args.device_id)
            print('GPU')
        else:
            self.dtype = torch.FloatTensor
            self.device = torch.device("cpu")
            print('CPU')
        # Setup tensorboard
        self.use_tb = self.args.exp_dir is not None
        self.log_dir = self.args.exp_dir


    # train validate
    # def train_validate(model, loader, optimizer, is_train, epoch, use_cuda):
    def train_validate(self,model, optimizer, is_train, epoch, use_cuda): 
        loss_func = contrastive_loss(tau=self.args.tau)
        # data_loader = loader.train_loader if is_train else loader.test_loader
        data_loader = self.train_loader if is_train else self.test_loader 

        if is_train:
            model.train()
            model.zero_grad()
        else:
            model.eval()

        desc = 'Train' if is_train else 'Validation'

        total_loss = 0.0

        tqdm_bar = tqdm(data_loader)
        print(f'len(tqdm_bar):{len(data_loader)}')
        for i, (x_i, _, x_j, _) in enumerate(tqdm_bar):   batch_idx, (data, target, batch_view_1, batch_view_2)
            x_i = x_i.cuda() if self.use_cuda else x_i
            x_j = x_j.cuda() if self.use_cuda else x_j

            x_i = x_i.unsqueeze(3)
            x_j = x_j.unsqueeze(3)

            x_i = x_i.float().to(self.device) 
            x_j = x_j.float().to(self.device)  

            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = loss_func(z_i, z_j)
            loss /= self.args.accumulation_steps

            if is_train:
                loss.backward()

            if (i + 1) % self.args.accumulation_steps == 0 and is_train:
                optimizer.step()
                model.zero_grad()

            total_loss += loss.item()

            tqdm_bar.set_description('{} Epoch: [{}] Loss: {:.4f}'.format(desc, epoch, loss.item()))

        return total_loss / (len(data_loader.dataset))

    def execute_graph(self,model, optimizer, scheduler, epoch, use_cuda): 
        t_loss = self.train_validate(model, optimizer, True, epoch, use_cuda) 
        v_loss = self.train_validate(model, optimizer, False, epoch, use_cuda) 

        scheduler.step(v_loss)

        if self.use_tb:
            self.logger.add_scalar(self.log_dir + '/train-loss', t_loss, epoch)
            self.logger.add_scalar(self.log_dir + '/valid-loss', v_loss, epoch)

        return v_loss


    def excute(self):
        # Setup asset directories
        if not os.path.exists('models'):
            os.makedirs('models')

        if not os.path.exists('runs'):
            os.makedirs('runs')

    # Logger
        if self.use_tb:
            self.logger = SummaryWriter(comment='_' + self.args.uid + '_' + self.args.dataset_name)

        if self.args.dataset_name == 'HAR':
            from .config_files.HAR_Configs import Config as Configs
            configs = Configs()
            in_channels = 9
            # valid_loader,
            self.train_loader, self.test_loader = data_generator('data/HAR', configs, 'self_supervised')
            print(f'len(train_loader):{len(self.train_loader)}')
            # print(f'len(valid_loader):{len(self.valid_loader)}')
            print(f'len(test_loader):{len(self.test_loader)}')

        elif self.args.dataset_name == 'wisdm':
            from .config_files.wisdm_Configs import Config as Configs
            configs = Configs()
            in_channels = 3
            train_loader, valid_loader, test_loader = data_generator('data/wisdm', configs, 'self_supervised')
            print(f'len(train_loader):{len(self.train_loader)}')
            print(f'len(valid_loader):{len(self.valid_loader)}')
            print(f'len(test_loader):{len(self.test_loader)}')

        elif self.args.dataset_name == 'epilepsy':
            from .config_files.epilepsy_Configs import Config as Configs
            configs = Configs()
            in_channels = 1
            train_loader, valid_loader, test_loader = data_generator('data/epilepsy', configs, 'self_supervised')
            print(f'len(train_loader):{len(self.train_loader)}')
            print(f'len(valid_loader):{len(self.valid_loader)}')
            print(f'len(test_loader):{len(self.test_loader)}')

        elif self.args.dataset_name == 'SHAR':
            from .config_files.SHAR_Configs import Config as Configs
            configs = Configs()
            in_channels = 3
            train_loader, valid_loader, test_loader = data_generator('data/SHAR', configs, 'self_supervised')
            print(f'len(train_loader):{len(train_loader)}')
            print(f'len(valid_loader):{len(valid_loader)}')
            print(f'len(test_loader):{len(test_loader)}')

       
        model = resnet50_cifar(self.args.feature_size, self.args.dataset_name).type(self.dtype) -add dataset_name
        # resnet50 input_dim=2048 because-- Bottleneck:expansion=4 -- 2048=512*4
        # model = resnet18_cifar(args.feature_size, args.dataset_name).type(dtype)-add dataset_name # try resnet18_cifar later
        # resnet18 input_dim=512  because-- BasicBlock:expansion=1 -- 512=512*1

        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.decay_lr)
        scheduler = ExponentialLR(optimizer, gamma=self.args.decay_lr)

        # Main training loop
        best_loss = np.inf

        # Resume training
        if self.args.load_model is not None:
            if os.path.isfile(self.args.load_model):
                checkpoint = torch.load(self.args.load_model)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                best_loss = checkpoint['val_loss']
                epoch = checkpoint['epoch']
                print('Loading model: {}. Resuming from epoch: {}'.format(self.args.load_model, epoch))
            else:
                print('Model: {} not found'.format(self.args.load_model))

        for epoch in range(self.args.epochs):
            # v_loss = execute_graph(model, loader, optimizer, scheduler, epoch, use_cuda)
            v_loss = self.execute_graph(model, optimizer, scheduler, epoch, self.use_cuda) 
            if v_loss < best_loss:
                best_loss = v_loss
                print('Writing model checkpoint')
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_loss': v_loss
                }
                t = time.localtime()
                timestamp = time.strftime('%b-%d-%Y_%H%M', t)
                file_name = 'models/{}_{}_{}_{:04.4f}.pt'.format(timestamp, self.args.uid, epoch, v_loss)

                torch.save(state, file_name)

        # TensorboardX logger
        self.logger.close()

        # save model / restart training


