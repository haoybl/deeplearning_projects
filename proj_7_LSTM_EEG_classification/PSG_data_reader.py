import h5py
import numpy as np


def prepare_train_valid_data():
    # Read train mat
    with h5py.File('train_data_PSG.mat', 'r') as f:
        train_data = f.get('train_data_collect/data')
        train_data = np.array(train_data).transpose([2, 1, 0])
        # print("Type: " + str(type(data)))
        print(train_data.shape)
        # print(data[0, 0:40, 0])
        # print(data[0, 3800:, 0])

        train_label = f.get('train_data_collect/label')
        train_label = np.array(train_label).transpose([1, 0])
        # print("Type: " + str(type(train_label)))
        print(train_label.shape)
        # print(train_label[0:40, 0:])
        # print(train_label[3800:, 0:])
        print("Train Raw Data/Label Read.")

    # Read Validation Data
    with h5py.File('valid_data_PSG.mat', 'r') as f:
        valid_data = f.get('valid_data_collect/data')
        valid_data = np.array(valid_data).transpose([2, 1, 0])
        print(valid_data.shape)

        valid_label = f.get('valid_data_collect/label')
        valid_label = np.array(valid_label).transpose([1, 0])
        print(valid_label.shape)

        print("Validation Raw Data/Label Read.")

    return train_data, train_label, valid_data, valid_label


def prepare_test_data():

    # Read test mat
    with h5py.File('test_data_PSG.mat', 'r') as f:
        test_data = f.get('test_data_collect/data')
        test_data = np.array(test_data).transpose([2, 1, 0])
        # print("Type: " + str(type(data)))
        print(test_data.shape)
        # print(data[0, 0:40, 0])
        # print(data[0, 3800:, 0])

        test_label = f.get('test_data_collect/label')
        test_label = np.array(test_label).transpose([1, 0])
        # print("Type: " + str(type(test_label)))
        print(test_label.shape)
        # print(test_label[0:40, 0:])
        # print(test_label[3800:, 0:])
        print("test Raw Data/Label Read.")

    return test_data, test_label



def next_batch(data, label, batch_size):

    num_sample = data.shape[0]
    perm = np.arange(num_sample)
    np.random.shuffle(perm)

    return data[perm[:batch_size]], label[perm[:batch_size]]