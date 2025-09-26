import argparse
import pandas as pd
import torch
import numpy as np

from utils.utils_basic import (
    set_device,
    set_device_for_tensor
)


# convert one long sequence into pairs of history and prediction sequences
def seq2instance(data, num_his, num_pred, device):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims, device=set_device(device))
    y = torch.zeros(num_sample, num_pred, dims, device=set_device(device))
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y


def load_speed_inputs(args: argparse.ArgumentParser):
    df = pd.read_hdf(args.speed_file)
    speed_data = torch.from_numpy(df.values.astype(np.float32))
    return df, speed_data


class DataSet:
    def __init__(self, df: pd.DataFrame, speed_data: torch.Tensor, args: argparse.ArgumentParser):
        self.df = df
        self.speed_data = speed_data
        self.args = args
        self.std = None
        self.mean = None

        self.train_steps = None
        self.val_steps = None
        self.test_steps = None
        self.SE = None
        self.TE = None
        self.device = set_device(args.device)

    def set_normalize_stats(self, speed_data: torch.Tensor):
        mean = torch.mean(speed_data)
        std = torch.std(speed_data)
        self.mean = mean
        self.std = std
        return speed_data

    def normalize_speed_data(self):
        mean, std = self.mean, self.std
        self.speed_data = (self.speed_data - mean) / std

    def set_devices(self):
        device = self.device
        self.speed_data = set_device_for_tensor(self.speed_data, device)
        self.SE = set_device_for_tensor(self.SE, device)
        self.TE = set_device_for_tensor(self.TE, device)

    def set_train_steps(self):
        num_step = self.df.shape[0]
        train_steps = round(self.args.train_ratio * num_step)
        test_steps = round(self.args.test_ratio * num_step)
        val_steps = num_step - train_steps - test_steps
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.test_steps = test_steps

    def get_train_data(self):
        train_steps = self.train_steps
        args, speed_data = self.args, self.speed_data
        SE, TE = self.SE, self.TE
        trainX = speed_data[: train_steps]
        trainX, trainY = seq2instance(trainX, args.num_his, args.num_pred, self.device)
        print("trainX.shape", trainX.shape)
        print("trainY.shape", trainY.shape)
        # get time embeddings
        trainTE = TE[:train_steps]
        print("trainTE.shape", trainTE.shape)
        # [num_sample, num_his + num_pred, 2]
        num_his = args.num_his
        num_pred = args.num_pred
        # set device
        trainX = set_device_for_tensor(trainX)
        trainY = set_device_for_tensor(trainY)
        trainTE = set_device_for_tensor(trainTE)
        SE = set_device_for_tensor(SE)
        return trainX, trainY, trainTE, SE
    
    def get_val_data(self):
        train_steps, val_steps = self.train_steps, self.val_steps
        args, speed_data= self.args, self.speed_data
        SE, TE = self.SE, self.TE
        valX = speed_data[train_steps: train_steps + val_steps]
        valX, valY = seq2instance(valX, args.num_his, args.num_pred, self.device)
        # get time embeddings
        valTE = TE[train_steps: train_steps + val_steps]
        # set device
        valX = set_device_for_tensor(valX)
        valY = set_device_for_tensor(valY)
        valTE = set_device_for_tensor(valTE)
        SE = set_device_for_tensor(SE)
        return valX, valY, valTE, SE
    
    def get_test_data(self):
        train_steps, val_steps, test_steps = self.train_steps, self.val_steps, self.test_steps
        args, speed_data = self.args, self.speed_data
        SE, TE = self.SE, self.TE
        testX = speed_data[train_steps + val_steps: train_steps + val_steps + test_steps]
        testX, testY = seq2instance(testX, args.num_his, args.num_pred, self.device)
        # get time embeddings
        testTE = TE[train_steps + val_steps: train_steps + val_steps + test_steps]
        # set device
        testX = set_device_for_tensor(testX)
        testY = set_device_for_tensor(testY)
        testTE = set_device_for_tensor(testTE)
        SE = set_device_for_tensor(SE)
        return testX, testY, testTE, SE
    
    def get_stats(self):
        return self.mean, self.std, self.SE


def load_data(args: argparse.ArgumentParser):
    # create dataset
    df, speed_data = load_speed_inputs(args)
    dataset = DataSet(df, speed_data, args)
    # load embeddings
    SE = load_SE_input(args)
    TE = load_TE_input(df, args)
    dataset.SE = SE
    dataset.TE = TE
    dataset.set_devices()
    # train/val/test
    dataset.set_train_steps()
    trainX, __, __, __ = dataset.get_train_data()
    # speed normalization
    dataset.set_normalize_stats(trainX)
    dataset.normalize_speed_data()
    return dataset


def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./figure/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))


def load_SE_input(args: argparse.ArgumentParser):
    with open(args.SE_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32, device=set_device(args.device))
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    return SE


def load_TE_input(df: pd.DataFrame, args: argparse.ArgumentParser):
    # time: [T, 2] <== [52116, 2]
    time = pd.DatetimeIndex(df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    total_secs = time.hour * 3600 + time.minute * 60 + time.second
    time_arr = df.reset_index()["index"]
    time_diff = (time_arr.iloc[1] - time_arr.iloc[0]).seconds
    timeofday = total_secs // time_diff
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)
    # [num_sample, num_his + num_pred, 2]
    TE_X, TE_Y = seq2instance(time, args.num_his, args.num_pred, args.device)
    TE = torch.cat((TE_X, TE_Y), 1).type(torch.int32)
    return TE


def load_model(args: argparse.ArgumentParser, model = None):
    if model is None:
        model = torch.load(
            args.model_file, 
            weights_only=False, 
            map_location=args.device
        )
        return model
    # otherwise, load weights only
    state_dict = torch.load(
        args.model_file, 
        weights_only=True, 
        map_location=args.device
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(args.device)
    return model
