import os
from shutil import copyfile

import torch 
import numpy as np 
import pandas as pd 
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

