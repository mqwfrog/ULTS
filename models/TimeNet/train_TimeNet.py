import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from framework import TimeNet
from ..data_loader.dataset_loader import data_generator

class train_TimeNet():

    def __init__(self, parser):
        self.args = parser.parse_args()
        self.log_dir = self.args.exp_dir
        self.dataset_name = self.args.dataset_name
        self.embeddings_dim = self.args.feature_size
        self.num_layers =  self.args.layers
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.lr
        self.epochs = self.args.epochs

    def excute(self):
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

        enc = TimeNet(self.embeddings_dim, num_layers=self.num_layers, dropout=self.dropout)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(enc.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=10)

        for epoch in range(self.epochs):
            enc.train()
            for data in train_loader:
                optimizer.zero_grad()
                outputs = enc(data.unsqueeze(2))  # Add a dimension for time steps
                loss = criterion(outputs, data.unsqueeze(2))
                loss.backward()
                optimizer.step()

        torch.save(enc.state_dict(), 'timenet_model.pth')



