import torch
import random
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy
import matplotlib.pyplot as plt   

def plot_confusion_matrix(cm, labels_name, title): 
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  
    plt.title(title)  
    plt.colorbar() 
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)   
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for first_index in range(len(cm)): 
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[first_index][second_index])

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path, data_type):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)

    ### plot confusion matrix
    print('Starting to plot confusion matrix...')
    dataset_name = data_type
    if dataset_name == 'sleepEDF':
        labels_name = ['Wake (W)', 'Non-rapid eye movement (N1)', 'Non-rapid eye movement (N2)',
                       'Non-rapid eye movement (N3)', 'Rapid Eye Movement (REM)']  # sleepEDF
    elif dataset_name == 'HAR':
        labels_name = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']  # HAR
    elif dataset_name == 'epilepsy':
        labels_name = ['epilepsy', 'not epilepsy'] # epilepsy
    elif dataset_name == 'SHAR':
        labels_name = ['StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping', 'GoingDownS', 'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight', 'FallingBack', 'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft']   # SHAR
    elif dataset_name == 'wisdm':
        labels_name = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs'] # wisdm
    elif dataset_name == 'DuckDuckGeese':
        labels_name = ['Black-bellied Whistling Duck', 'Canadian Goose', 'Greylag Goose', 'Pink Footed Goose',
                       'White-faced Whistling Duck']
    elif dataset_name == 'FingerMovements':
        labels_name = ['Left', 'Right']
    elif dataset_name == 'PenDigits':
        labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif dataset_name == 'PhonemeSpectra':
        labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19',
                       '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
                       '36', '37', '38']
    elif dataset_name == 'StandWalkJump':
        labels_name = ['Standing', 'Walking', 'Jumping']



    plot_confusion_matrix(cm, labels_name, f"{dataset_name}--- Confusion Matrix")
    plt.subplots_adjust(bottom=0.15)
    if not os.path.exists(f'./cm/cm_{dataset_name}'):
        os.makedirs(f'./cm/cm_{dataset_name}')
    plt.savefig(f'./cm/cm_{dataset_name}/{dataset_name}_cm.png', format='png', bbox_inches='tight')
    plt.show()



def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("train_TSTCC.py", os.path.join(destination_dir, "train_TSTCC.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))
