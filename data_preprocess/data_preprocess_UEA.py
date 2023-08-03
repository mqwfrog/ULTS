import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sktime.utils.data_io import load_from_arff_to_dataframe
from sklearn.preprocessing import LabelEncoder

# original UEA(0,1,2) [instances, length, features/channels]
# UEA(0,1,2) --> later will be permuted in dataloader-->get UEA(0,2,1) [instances, features/channels, length]

DATA_DIR = '../data'
def mkdir_if_not_exists(loc, file=False):
    loc_ = os.path.dirname(loc) if file else loc
    if not os.path.exists(loc):
        os.makedirs(loc_, exist_ok=True)

def create_torch_data(train_file, test_file):
    # Get arff format
    train_data, train_labels = load_from_arff_to_dataframe(train_file)
    test_data, test_labels = load_from_arff_to_dataframe(test_file)

    def convert_data(data):
        # Expand the series to numpy
        data_expand = data.applymap(lambda x: x.values).values
        # Single array, then to tensor
        data_numpy = np.stack([np.vstack(x).T for x in data_expand])
        tensor_data = torch.Tensor(data_numpy)
        return tensor_data

    train_data, test_data = convert_data(train_data), convert_data(test_data)

    # Encode labels as often given as strings
    encoder = LabelEncoder().fit(train_labels)
    train_labels, test_labels = encoder.transform(train_labels), encoder.transform(test_labels)
    train_labels, test_labels = torch.Tensor(train_labels), torch.Tensor(test_labels)

    return train_data, test_data, train_labels, test_labels
def save_pickle(obj, filename, protocol=4, create_folder=True):
    if create_folder:
        mkdir_if_not_exists(filename, file=True)

    # Save
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol)

def convert_all_files(dataset='uea'):
    """ Convert all files from a given /raw/{subfolder} into torch data to be stored in /interim. """
    assert dataset in ['uea', 'ucr']
    folder = 'UEA'
    arff_folder = DATA_DIR + '/raw/{}/Multivariate_arff'.format(folder)

    # Time for a big for loop
    for ds_name in tqdm([x for x in os.listdir(arff_folder) if os.path.isdir(arff_folder + '/' + x)]):
        # File locations
        print(f'ds_name:{ds_name}')
        train_file = arff_folder + '/{}/{}_TRAIN.arff'.format(ds_name, ds_name)
        test_file = arff_folder + '/{}/{}_TEST.arff'.format(ds_name, ds_name)

        # Ready save dir
        save_dir = DATA_DIR + '/processed/{}/{}'.format(folder, ds_name)
        print(f'save_dir:{save_dir}')
        # If files don't exist, skip.
        if any([x.split('/')[-1] not in os.listdir(arff_folder + '/{}'.format(ds_name)) for x in (train_file, test_file)]):
            if ds_name not in ['Images', 'Descriptions']:
                print('No files found for folder: {}'.format(ds_name))
            continue
        # elif os.path.isdir(save_dir):
        #     print('Files already exist for: {}'.format(ds_name))
        #     continue
        else:
            train_data, test_data, train_labels, test_labels = create_torch_data(train_file, test_file)

            dat_dict = dict()
            dat_dict["samples"] = train_data
            dat_dict["labels"] = train_labels
            torch.save(dat_dict, ds_name+"_train.pt")

            dat_dict = dict()
            dat_dict["samples"] = test_data
            dat_dict["labels"] = test_labels
            torch.save(dat_dict, ds_name+"_test.pt")

            # # Compile train and test data together
            # data = torch.cat([train_data, test_data])
            # labels = torch.cat([train_labels, test_labels])
            #
            # # Save original train test indexes in case we wish to use original splits
            # original_idxs = (np.arange(0, train_data.size(0)), np.arange(train_data.size(0), data.size(0)))

            # # Save data
            # save_pickle(data, save_dir + '/data.pkl')
            # save_pickle(labels, save_dir + '/labels.pkl')
            # save_pickle(original_idxs, save_dir + '/original_idxs.pkl')


if __name__ == '__main__':
    dataset = 'uea'
    convert_all_files(dataset)
