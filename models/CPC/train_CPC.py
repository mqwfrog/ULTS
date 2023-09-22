import argparse
import torch
from model import CPCModel as cpcmodel
from config import configs
from feature_loader import getEncoder, getAR, loadArgs
from ..data_loader.dataset_loader import data_generator 

class train_CPC():
    
    def __init__(self,parser):
        self.args = parser.parse_args()
        self.dataset_name  = self.args.dataset_name
        self.method = 'CPC'
        self.training_mode = self.args.training_mode
        self.data_dir = self.args.data_dir

    def excute(self):
        locArgs = configs()
        loadArgs(locArgs, self.args)
        encoderNet = getEncoder(locArgs)
        arNet = getAR(locArgs)
        model = cpcmodel(encoderNet, arNet)
        train_dl, valid_dl, test_dl = data_generator(os.path.join(self.data_dir, self.dataset_name), locArgs, self.training_mode) 

        model.train(train_dl)
