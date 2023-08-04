import os
import json
import math
import torch
import numpy
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader

import scikit_wrappers


class train_TLoss():
     def __init__(self,parser):
        self.args=parser.parse_args()
        self.dataset = self.args.dataset_name
        self.path = self.args.data_dir
        self.save_path = self.args.exp_dir
        self.cuda = not self.args.no_cuda
        self.gpu = self.args.device_id
        self.hyper = self.args.hyper 
        self.load = self.args.load
        self.fit_classifier = self.args.fit_classifier

    def fit_hyperparameters(file, train, train_labels, cuda, gpu,
                        save_memory=False):
        classifier = scikit_wrappers.CausalCNNEncoderClassifier()

        # Loads a given set of hyperparameters and fits a model with those
        hf = open(os.path.join(file), 'r')
        params = json.load(hf)
        hf.close()
        # Check the number of input channels
        params['in_channels'] = numpy.shape(train)[1]
        params['cuda'] = cuda
        params['gpu'] = gpu
        classifier.set_params(**params)
        return classifier.fit(
            train, train_labels, save_memory=save_memory, verbose=True
        )

    def excute(self):
        train_dataset = torch.load(os.path.join(self.path, "train.pt"))
        # valid_dataset = torch.load(os.path.join(self.path, "val.pt"))
        test_dataset = torch.load(os.path.join(self.path, "test.pt"))
        train = np.array(train_dataset["samples"])
        train = train.astype(float)
        train_labels = np.array(train_dataset["labels"])
        test = np.array(test_dataset["samples"])
        test = test.astype(float)
        test_labels = np.array(test_dataset["labels"])

        if self.dataset not in ('HAR','wisdm','SHAR','epilepsy'):
            print('UEA-----need permute')
            train = np.swapaxes(train, 1, 2)  
            test = np.swapaxes(test, 1, 2)
      
        print(f'self.load:{self.load}')#False-training, True-load
        print(f'self.fit_classifier:{self.fit_classifier}')#True-load the model and retrain the classifier
        if not self.load and not self.fit_classifier:
            classifier = fit_hyperparameters(
                self.hyper, train, train_labels, self.cuda, self.gpu,
                save_memory=True
            )
        else:
            classifier = scikit_wrappers.CausalCNNEncoderClassifier()
            hf = open(
                os.path.join(
                    self.save_path, self.dataset + '_hyperparameters.json'
                ), 'r'
            )
            hp_dict = json.load(hf)
            hf.close()
            hp_dict['cuda'] = self.cuda
            hp_dict['gpu'] = self.gpu
            classifier.set_params(**hp_dict)
            classifier.load(os.path.join(self.save_path, self.dataset))

        if not self.load:#load=false
            if self.fit_classifier:#fit_classifier=true
                classifier.fit_classifier(classifier.encode(train), train_labels)
            classifier.save(
                os.path.join(self.save_path, self.dataset)
            )
            with open(
                os.path.join(
                    self.save_path, self.dataset + '_hyperparameters.json'
                ), 'w'
            ) as fp:
                json.dump(classifier.get_params(), fp)
        print("Test accuracy: " + str(classifier.mqw_score(test, test_labels, self.dataset)))dataset_name
