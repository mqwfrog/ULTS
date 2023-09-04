import os
import numpy as np
import torch
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..data_loader.dataset_loader import data_generator 

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA_origin(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    print(train_X.shape) #[40,100,6] [Train Size, Length, channels/variables]
    print(train_X)
    print(train_y.shape) #[40,]
    print(train_y) #[2 2 2 2 1 1 3 0.......]
    return train_X, train_y, test_X, test_y


def load_UEA(dataset):
    print('************UEA archive************')
    # UEA archive
    print(f'dataset:{dataset}')
    train_data = torch.load(f'datasets/{dataset}/train.pt')
    test_data = torch.load(f'datasets/{dataset}/test.pt')
    # dat_dict = dict()
    train_X = train_data["samples"].numpy()
    train_y = train_data["labels"].numpy()
    test_X = test_data["samples"].numpy()
    test_y = test_data["labels"].numpy()

    print(f'train_X.shape:{train_X.shape}')  # BasicMotions[40,100,6] [Train Size, Length, channels/variables] HAR[5881,128,9]
    print(f'test_X.shape:{test_X.shape}')
    # print(train_X)
    print(f'train_y.shape:{train_y.shape}')    # BasicMotions[40,] HAR[5881,]
    # print(train_y)  # BasicMotions[2 2 2 2 1 1 3 0.......] HAR[5. 1. 1. ... 0. 4. 0.]
    return train_X, train_y, test_X, test_y

def load_MTS(dataset):
    print('************HAR / SHAR / wisdm************')
    # MTS: multivariate time series
    # datasets: HAR / SHAR / wisdm
    # train_data = torch.load(f'datasets/{dataset}/train.pt')
    # test_data = torch.load(f'datasets/{dataset}/test.pt')
    # from config_files.HAR_Configs import Config as Configs 
    # configs = Configs() 
    # train_data , valid_dl, test_data = data_generator('./datasets/HAR', configs, 'self_supervised')  # train_linear
    print(f'dataset:{dataset}')
    train_data = torch.load(f'datasets/{dataset}/train.pt')
    test_data = torch.load(f'datasets/{dataset}/test.pt')
    # dat_dict = dict()
    train_X = train_data["samples"].numpy()
    train_y = train_data["labels"].numpy()
    test_X = test_data["samples"].numpy()
    test_y = test_data["labels"].numpy()


    print(f'train_X.shape:{train_X.shape}')  # BasicMotions[40,100,6] [Train Size, Length, channels] HAR[5881,128,9]
    print(f'test_X.shape:{test_X.shape}')
    # print(train_X)
    print(f'train_y.shape:{train_y.shape}')    # BasicMotions[40,] HAR[5881,]
    # print(train_y)  # BasicMotions[2 2 2 2 1 1 3 0.......] HAR[5. 1. 1. ... 0. 4. 0.]
    return train_X, train_y, test_X, test_y

def load_UTS(dataset):
    # UTS: Univariate time series
    # datasets: epilepsy / sleepEDF
    print(f'dataset:{dataset}')
    train_data = torch.load(f'datasets/{dataset}/train.pt')
    test_data = torch.load(f'datasets/{dataset}/test.pt')
    # dat_dict = dict()
    train_X = train_data["samples"].numpy()
    train_y = train_data["labels"].numpy()
    test_X = test_data["samples"].numpy()
    test_y = test_data["labels"].numpy()


    train = train_X[:, 1:].astype(np.float64)
    train_labels = np.vectorize(train_y[:, 0])
    test = test_X[:, 1:].astype(np.float64)
    test_labels = np.vectorize(test_y[:, 0])

    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0

def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)

def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols
