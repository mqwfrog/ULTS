import os
import random
import re
import torch
import numpy as np
import pandas as pd
import pickle
import scipy
import scipy.interpolate
import pathlib
import typing
import errno
from pathlib import Path
from typing import Optional, Union, List, Tuple
from sklearn.model_selection import train_test_split

__all__ = ['WISDM', 'load', 'load_raw']

# Meta Info
SUBJECTS = tuple(range(1, 36+1))
ACTIVITIES = tuple(['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs'])
Sampling_Rate = 20 # Hz

class BaseDataset(object):
    def __init__(self, path: Path):
        self.path = path
    def load(self, *args):
        raise NotImplementedError

class WISDM(BaseDataset):
    def __init__(self, path:Path):
        super().__init__(path)

    def load(self, window_size:int, stride:int, ftrim_sec:int=3, btrim_sec:int=3, subjects:Optional[list]=None) -> Tuple[np.ndarray, np.ndarray]:
        segments, meta = load(path=self.path)
        segments = [m.join(seg) for seg, m in zip(segments, meta)]

        x_frames, y_frames = [], []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                return_error_value=None)
            if fs is not None:
                x_frames += [fs[:, :, 3:]]
                y_frames += [np.uint8(fs[:, 0, 1:2][..., ::-1])]
            else:
                pass
        x_frames = np.concatenate(x_frames).transpose([0, 2, 1])
        y_frames = np.concatenate(y_frames)
        y_frames -= np.min(y_frames)
        y_frames = y_frames.squeeze(1)

        # subject filtering
        if subjects is not None:
            flags = np.zeros(len(x_frames), dtype=bool)
            for sub in subjects:
                flags = np.logical_or(flags, y_frames[:, 1] == sub)
            x_frames = x_frames[flags]
            y_frames = y_frames[flags]

        return x_frames, y_frames

def check_path(path: Union[Path, str]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise TypeError('expected type of "path" is Path or str, but {}'.format(type(path)))
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    return path

def to_frames_using_reshape(src: np.ndarray, window_size: int) -> np.ndarray:
    num_frames = (src.shape[0] - window_size) // window_size + 1
    ret = src[:(num_frames * window_size)]
    return ret.reshape(-1, window_size, *src.shape[1:])

def to_frames(src: np.ndarray, window_size: int, stride: int, stride_mode: str = 'index') -> np.ndarray:
    assert stride > 0, 'stride={}'.format(stride)
    assert stride_mode in ['index', 'nptrick'], "stride_mode is 'index' or 'nptrick'. stride_mode={}".format(
        stride_mode)
    if stride == window_size:
        return to_frames_using_reshape(src, window_size)
    elif stride_mode == 'index':
        return to_frames_using_index(src, window_size, stride)
    else:
        return to_frames_using_nptricks(src, window_size, stride)

def to_frames_using_index(src: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    assert stride > 0, 'stride={}'.format(stride)
    num_frames = (len(src) - window_size) // stride + 1
    idx = np.arange(window_size).reshape(-1, window_size).repeat(num_frames, axis=0) + np.arange(
        num_frames).reshape(num_frames, 1) * stride
    return src[idx]

def to_frames_using_nptricks(src: np.ndarray, window_size: int, stride: int) -> np.ndarray:

    assert stride > 0, 'stride={}'.format(stride)
    num_frames = (src.shape[0] - window_size) // stride + 1
    ret_shape = (num_frames, window_size, *src.shape[1:])
    strides = (stride * src.strides[0], *src.strides)
    return np.lib.stride_tricks.as_strided(src, shape=ret_shape, strides=strides)

def split_using_sliding_window(segment: np.ndarray, **options) -> np.ndarray:
    assert len(segment.shape) == 2, "Segment's shape is (segment_size, ch). This segment shape is {}".format(
        segment.shape)
    window_size = options.pop('window_size', 512)
    stride = options.pop('stride', None)
    ftrim = options.pop('ftrim', 5)
    btrim = options.pop('btrim', 5)
    return_error_value = options.pop('return_error_value', None)
    assert not bool(options), "args error: key {} is not exist.".format(list(options.keys()))
    assert type(window_size) is int, "type(window_size) is int: {}".format(type(window_size))
    assert ftrim >= 0 and btrim >= 0, "ftrim >= 0 and btrim >= 0: ftrim={}, btrim={}".format(ftrim, btrim)
    if type(segment) is not np.ndarray:
        return return_error_value
    if len(segment) < ftrim + btrim:
        return return_error_value
    if btrim == 0:
        seg = segment[ftrim:].copy()
    else:
        seg = segment[ftrim: -btrim].copy()
    if len(seg) < window_size:
        return return_error_value
    if stride is None:
        stride = window_size
    return to_frames(seg, window_size, stride, stride_mode='index')

def split_using_target(src: np.ndarray, target: np.ndarray) -> typing.Dict[int, typing.List[np.ndarray]]:
    from collections import defaultdict
    rshifted = np.roll(target, 1)
    diff = target - rshifted
    diff[0] = 1
    idxes = np.where(diff != 0)[0]

    ret = defaultdict(list)
    for i in range(1, len(idxes)):
        ret[target[idxes[i - 1]]].append(src[idxes[i - 1]:idxes[i]].copy())
    ret[target[idxes[-1]]].append(src[idxes[-1]:].copy())
    return dict(ret)

def interpolate(src: np.ndarray, rate: int, kind: str = 'linear', axis: int = -1) -> np.ndarray:
    N = src.shape[axis]
    x_low = np.linspace(0, 1, N)
    x_target = np.linspace(0, 1, N + (N - 1) * (rate - 1))
    f = scipy.interpolate.interp1d(x_low, src, kind=kind, axis=axis)
    return f(x_target)

def pickle_dump(obj: typing.Any, path: typing.Union[str, pathlib.Path]) -> None:
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)
    return

def pickle_load(path: pathlib.Path) -> typing.Any:
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data

def load(path:Union[Path,str]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    path = check_path(path)
    raw = load_raw(path)
    data, meta = reformat(raw)
    return data, meta

def load_raw(path:Path) -> pd.DataFrame:
    path = path / 'WISDM_ar_v1.1_raw.txt'
    with path.open('r') as fp:
        whole_str = fp.read()

    whole_str = whole_str.replace(',;', ';')
    semi_separated = re.split('[;\n]', whole_str)
    semi_separated = list(filter(lambda x: x != '', semi_separated))
    comma_separated = [r.strip().split(',') for r in semi_separated]

    # debug
    for s in comma_separated:
        if len(s) != 6:
            print('[miss format?]: {}'.format(s))

    raw_data = pd.DataFrame(comma_separated)
    raw_data.columns = ['user', 'activity', 'timestamp', 'x-acceleration', 'y-acceleration', 'z-acceleration']
    raw_data['z-acceleration'] = raw_data['z-acceleration'].replace('', np.nan)

    # convert activity name to activity id
    raw_data = raw_data.replace(list(ACTIVITIES), list(range(len(ACTIVITIES))))

    raw_data = raw_data.astype({'user': 'uint8', 'activity': 'uint8', 'timestamp': 'uint64', 'x-acceleration': 'float64', 'y-acceleration': 'float64', 'z-acceleration': 'float64'})
    raw_data[['x-acceleration', 'y-acceleration', 'z-acceleration']] = raw_data[['x-acceleration', 'y-acceleration', 'z-acceleration']].fillna(method='ffill')

    return raw_data


def reformat(raw) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    raw_array = raw.to_numpy()
    
    # segment (by user and activity)
    sdata_splited_by_subjects = split_using_target(src=raw_array, target=raw_array[:, 0])
    segments = []
    for sub_id in sdata_splited_by_subjects.keys():
        for src in sdata_splited_by_subjects[sub_id]:
            splited = split_using_target(src=src, target=src[:, 1])
            for act_id in splited.keys():
                segments += splited[act_id]

    segments = list(map(lambda seg: pd.DataFrame(seg, columns=raw.columns).astype(raw.dtypes.to_dict()), segments))
    data = list(map(lambda seg: pd.DataFrame(seg.iloc[:, 3:], columns=raw.columns[3:]), segments))
    meta = list(map(lambda seg: pd.DataFrame(seg.iloc[:, :3], columns=raw.columns[:3]), segments))

    return data, meta 

if __name__ == '__main__':

    output_dir = r'../../data/wisdm'

    wisdm_path = Path('./')
    wisdm = WISDM(wisdm_path)

    x, y = wisdm.load(window_size=256, stride=256, ftrim_sec=0, btrim_sec=0)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    train_sample_num = int(1.0 * len(X_train))  # select x% data
    train_sample_list = [i for i in range(len(X_train))]
    train_sample_list = random.sample(train_sample_list, train_sample_num)
    X_train = X_train[train_sample_list, :]
    y_train = y_train[train_sample_list]

    test_sample_num = int(1.0 * len(X_test))  #  select x% data
    test_sample_list = [i for i in range(len(X_test))]
    test_sample_list = random.sample(test_sample_list, test_sample_num)
    X_test = X_test[test_sample_list, :]
    y_test = y_test[test_sample_list]

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


