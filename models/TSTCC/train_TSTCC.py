import torch

import os
import numpy as np
from datetime import datetime
import argparse

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  

from config_files.HAR_Configs import Config
from utils import _logger, set_requires_grad
from ..data_loader.dataset_loader import data_generator 
from trainer import Trainer, model_evaluate
from TC import TC
from utils import _calc_metrics, copy_Files
from model import base_Model

class train_TSTCC():

    def __init__(self, parser):
        self.args = parser.parse_args()
        # Args selections
        self.start_time = datetime.now()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = self.args.dataset_name
        self.method = 'TSTCC'
        self.experiment_description = self.method
        self.training_mode = self.args.training_mode
        self.logs_save_dir = self.args.exp_dir
        # ##### random seeds for reproducibility ########
        self.SEED = self.args.seed

    def plot_embedding(self,data, label, title): 
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)	
        fig = plt.figure(figsize=(15, 10))
        ax = plt.subplot(111)		
        ### 2D
        if self.dataset_name == 'SHAR' or 'CharacterTrajectories':  
            for i in range(data.shape[0]): 
                plt.text(data[i, 0], data[i, 1], int(label[i]), color=plt.cm.tab20(label[i] / 20),
                         fontdict={'weight': 'bold', 'size': 7})
            plt.xticks()  
            plt.yticks()
            plt.title(title, fontsize=14)
        elif self.dataset_name == 'PhonemeSpectra': 
            for i in range(data.shape[0]): 
                plt.text(data[i, 0], data[i, 1], int(label[i]), color=plt.cm.tab20(label[i] / 20),
                         fontdict={'weight': 'bold', 'size': 7})

            plt.xticks() 
            plt.yticks()
            plt.title(title, fontsize=14)
        else:
            for i in range(data.shape[0]):  
                plt.text(data[i, 0], data[i, 1], int(label[i]), color=plt.cm.tab10(label[i] / 10),
                         fontdict={'weight': 'bold', 'size': 7})
            plt.xticks()  
            plt.yticks()
            plt.title(title, fontsize=14)
        return fig


    def excute(self):
        os.makedirs(self.logs_save_dir, exist_ok=True)

        exec(f'from config_files.{self.data_type}_Configs import Config')
        configs = Config()  

        print(self.SEED)
        torch.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.SEED)

        experiment_log_dir = os.path.join(self.logs_save_dir, self.experiment_description, self.training_mode + f"_seed_{self.SEED}")
        os.makedirs(experiment_log_dir, exist_ok=True)

        # loop through domains
        counter = 0
        src_counter = 0

        # Logging
        log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
        logger = _logger(log_file_name)
        logger.debug("=" * 45)
        logger.debug(f'Dataset: {self.data_type}')
        logger.debug(f'Method:  {self.method}')
        logger.debug(f'Mode:    {self.training_mode}')
        logger.debug("=" * 45)

        # Load datasets
        data_path = f"./data/{self.data_type}"
        # train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
        train_dl, test_dl = data_generator(data_path, configs, self.training_mode)
        # print(train_dl)
        logger.debug("Data loaded ...")

        # Load Model
        model = base_Model(configs).to(self.device)
        temporal_contr_model = TC(configs, self.device).to(self.device)

        if self.training_mode == "fine_tune":
        # load saved model of this experiment
            load_from = os.path.join(
                os.path.join(self.logs_save_dir, self.experiment_description, f"self_supervised_seed_{self.SEED}",
                             "saved_models"))
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=self.device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del pretrained_dict[i]
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if self.training_mode == "train_linear" or "tl" in self.training_mode:
            load_from = os.path.join(
                os.path.join(self.logs_save_dir, self.experiment_description, f"self_supervised_seed_{self.SEED}",
                             "saved_models"))
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=self.device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # delete these parameters (Ex: the linear layer at the end)
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                         del pretrained_dict[i]
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

        if self.training_mode == "random_init":
            model_dict = model.state_dict()

            # delete all the parameters except for logits
            del_list = ['logits']
            pretrained_dict_copy = model_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                     if j in i:
                        del model_dict[i]
            set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                               weight_decay=3e-4)
        temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                                        betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

        if self.training_mode == "self_supervised":  # to do it only once
            copy_Files(os.path.join(self.logs_save_dir, self.experiment_description), self.data_type)

            # Trainer
            # Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)
            Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, test_dl, self.device, logger,
                    configs, experiment_log_dir, self.training_mode)

            if self.training_mode != "self_supervised":
            # Testing
                outs = model_evaluate(model, temporal_contr_model, test_dl, self.device, self.training_mode)
            total_loss, total_acc, pred_labels, true_labels, representations = outs  # mqw
            _calc_metrics(pred_labels, true_labels, experiment_log_dir, self.args.home_path, self.data_type)

            ### TSNE Embeddings of representations
            print('Starting to compute t-SNE Embeddings...')
            ts = TSNE(n_components=2, init='pca',
                      random_state=0)  
            print(f'x.shape{representations.shape}')
            result = ts.fit_transform(representations)  
            fig = self.plot_embedding(result, true_labels,
                                 f't-SNE Embeddings of Time Series Representations---Dataset: {self.dataset_name}')  # 显示图像
            if not os.path.exists(f'./eb/eb_{self.dataset_name}'):
                os.makedirs(f'./eb/eb_{self.dataset_name}')
            plt.savefig(f'./eb/eb_{self.dataset_name}/{self.dataset_name}_pca_2d.png', format='png',
                        bbox_inches='tight')  # change-16 命名 init='pca'//'random'  以及2d还是3d
            plt.show()

            logger.debug(f"Training time is : {datetime.now() - self.start_time}")

