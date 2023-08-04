import os
import torch
import yaml
import argparse
from mlp_head import MLPHead
from resnet_base_network import ResNet18
from trainer import BYOLTrainer
from ..data_loader.dataset_loader import data_generator 

torch.manual_seed(0)

class train_BYOL():
    
    def __init__(self,parser):
        self.args = parser.parse_args()
        self.dataset_name  = self.args.dataset_name
        self.method = 'BYOL'
        self.training_mode = self.args.training_mode
        self.data_dir = self.args.data_dir


    def excute(self):
        config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Training with: {device}")

        from config_files.wisdm_Configs import Config as Configs  
        configs = Configs() 
        train_dl, valid_dl, test_dl = data_generator(os.path.join(self.data_dir, self.dataset_name), configs, self.training_mode)  # train_linear # change-2

        # online network
        online_network = ResNet18(**config['network']).to(device)
        pretrained_folder = config['network']['fine_tune_from']

        # load pre-trained model if defined
        if pretrained_folder:
            try:
                checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

                # load pre-trained parameters
                load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                         map_location=torch.device(torch.device(device)))

                online_network.load_state_dict(load_params['online_network_state_dict'])

            except FileNotFoundError:
                print("Pre-trained weights not found. Training from scratch.")

        # predictor network
        predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                            **config['network']['projection_head']).to(device)

        # target encoder
        target_network = ResNet18(**config['network']).to(device)
        # target_network = ResNet18(configs, **config['network']).to(device)

        optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                    **config['optimizer']['params'])

        trainer = BYOLTrainer(online_network=online_network,
                              target_network=target_network,
                              optimizer=optimizer,
                              predictor=predictor,
                              device=device,
                              **config['trainer'])

        trainer.train(train_dl)
