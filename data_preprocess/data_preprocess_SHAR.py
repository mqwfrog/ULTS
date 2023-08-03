import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from pathlib import Path
from typing import List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split
from data_preprocessing.base import BaseDataset, check_path

__all__ = ['UniMib', 'load', 'load_raw']

# Meta Info
SUBJECTS = tuple(range(1, 30+1))
ACTIVITIES = tuple(['StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping', 'GoingDownS', 'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight', 'FallingBack', 'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft'])
GENDER = {'M': 0, 'F': 1}
Sampling_Rate = 50 # Hz

class UniMib(BaseDataset):
    def __init__(self, path:Path):
        super().__init__(path)
    
    def load(self, data_type:str, window_size:Optional[int]=None, stride:Optional[int]=None, ftrim_sec:int=3, btrim_sec:int=3, subjects:Optional[list]=None) -> Tuple[np.ndarray, np.ndarray]:
        if data_type != 'raw':
            data, meta = load(path=self.path, data_type=data_type)
            segments = {'acceleration': data, 'activity': meta['activity'], 'subject': meta['subject']}
            x = np.stack(segments['acceleration']).transpose(0, 2, 1)
            y = np.stack(segments['activity'])
            x_frames = x
            y -= np.min(y)
            y_frames = y

        # subject filtering
        if subjects is not None:
            flags = np.zeros(len(x_frames), dtype=bool)
            for sub in subjects:
                flags = np.logical_or(flags, y_frames[:, 1] == sub)
            x_frames = x_frames[flags]
            y_frames = y_frames[flags]
        return x_frames, y_frames


def load(path:Union[Path,str], data_type:str='full') -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    path = check_path(path)
    raw = load_raw(path, data_type)
    data, meta = reformat(raw)
    return data, meta


def load_raw(path:Path, data_type:str = 'full') -> Union[Tuple[np.ndarray, pd.DataFrame], Tuple[List[pd.DataFrame], pd.DataFrame]]:
    if not isinstance(data_type, str):
        raise TypeError('expected type of "type" argument is str, but {}'.format(type(data_type)))
    if data_type not in ['full', 'adl', 'fall']:
        raise ValueError('unknown data type, {}'.format(data_type))
    if data_type == 'full':
        prefix = 'acc'
    elif data_type == 'adl':
        prefix = 'adl'
    elif data_type == 'fall':
        prefix = 'fall'
    if data_type != 'raw':
        data = loadmat(str(path / f'{prefix}_data.mat'))[f'{prefix}_data'].reshape([-1, 3, 151])
        labels = loadmat(str(path / f'{prefix}_labels.mat'))[f'{prefix}_labels']
        meta = labels
        meta = pd.DataFrame(meta, columns=['activity', 'subject', 'trial_id'])
        meta = meta.astype({'activity': np.int8, 'subject': np.int8, 'trial_id': np.int8})
    return data, meta


def reformat(raw) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    data, meta = raw
    data = list(map(lambda x: pd.DataFrame(x.T, columns=['x', 'y', 'z']), data))
    return data, meta


if __name__ == '__main__':

    output_dir = r'../../data/SHAR'
    unimib_path = Path('./data')
    unimib = UniMib(unimib_path)
    x, y = unimib.load(data_type='full')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_train)
    dat_dict["labels"] = torch.from_numpy(y_train)
    torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_val)
    dat_dict["labels"] = torch.from_numpy(y_val)
    torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_test)
    dat_dict["labels"] = torch.from_numpy(y_test)
    torch.save(dat_dict, os.path.join(output_dir, "test.pt"))
