import os
import sys
import random
import datetime
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

from functools import lru_cache

import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
import numpy as np

def save_dataframe(df:pd.DataFrame, path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl")->None:
    if format == "csv":
        df.to_csv(path)
    elif format ==  "pkl":
        df.to_pickle(path)
    elif format == "parquet":
        df.to_parquet(path)
    elif format == "feather":
        df.to_feather(path)
    else:
        raise NotImplementedError()

def load_dataframe(path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl")->pd.DataFrame:
    if format == "csv":
        df = pd.read_csv(path, index_col=0)
    elif format ==  "pkl":
        df = pd.read_pickle(path)
    elif format == "parquet":
        df = pd.read_parquet(path)
    elif format == "feather":
        df = pd.read_feather(path)
    else:
        raise NotImplementedError()
    return df

class StockDataset(Dataset):
    """继承自 Dataset，用于加载和预处理股票数据集。支持序列化分割和随机分割，并提供了数据集的信息。"""
    def __init__(self, 
                 data_x_dir:str, 
                 data_y_dir:str, 
                 label_name:str, 
                 format:Literal["csv", "pkl", "parquet", "feather"],
                 cache_size:int) -> None:
        super().__init__()
        self.data_x_dir:str = data_x_dir
        self.data_y_dir:str = data_y_dir
        
        self.format:str = format
        self.label_name:str = label_name

        self.x_file_paths:List[str] = [os.path.join(data_x_dir, f) for f in os.listdir(data_x_dir) if f.endswith(format)]
        self.y_file_paths:List[str] = [os.path.join(data_y_dir, f) for f in os.listdir(data_y_dir) if f.endswith(format)]

        self.cache_size:int = cache_size
        self.load_df:Callable = self.create_cached_loader(self.cache_size)
    
    def create_cached_loader(self, cache_size) -> Callable:
        @lru_cache(maxsize=cache_size)
        def load_dataframe(path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl")->pd.DataFrame:
            if format == "csv":
                df = pd.read_csv(path, index_col=0)
            elif format ==  "pkl":
                df = pd.read_pickle(path)
            elif format == "parquet":
                df = pd.read_parquet(path)
            elif format == "feather":
                df = pd.read_feather(path)
            else:
                raise NotImplementedError()
            return df
        return load_dataframe

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        """根据索引 index 从文件路径列表中读取对应的 CSV 文件。将读取的数据转换为 PyTorch 张量，并处理 NaN 值。
        注意，这是一种惰性获取数据的协议，数据集类并不事先将所有数据载入内存，而是根据索引再做读取。能较好地应对大量数据的情形，节约内存；但训练时由于IO瓶颈速度较慢，需要花费大量时间在csv的读取上面。"""
        X = self.load_df(self.x_file_paths[index], format=self.format).iloc[:,2:]   #pd.read_csv(self.x_file_paths[index], index_col=[0,1,2])
        X = torch.from_numpy(X.values).nan_to_num().float()
        y = self.load_df(self.y_file_paths[index], format=self.format).iloc[:,2:][self.label_name] #pd.read_csv(self.y_file_paths[index], index_col=[0,1,2])[self.label_name]
        y = torch.from_numpy(y.values).nan_to_num().float()
        return X, y
    
    def __len__(self):          
        """获取数据集长度"""
        return min(len(self.x_file_paths), len(self.y_file_paths))
    
    def info(self):
        """返回数据集的第一个样本的形状、标签形状和数据集长度。作为参考信息。"""
        x, y = self[0]
        x_shape = x.shape
        y_shape = y.shape
        return {"x_shape":x_shape, "y_shape":y_shape, "len":len(self)}
    
    def serial_split(self, ratios:List[Number], mask:int=0) -> List["StockDataset"]:
        """序列化分割。根据给定的比例 ratios 将数据集分割成多个子数据集，且保持原顺序。返回一个StockDataset类的列表。"""
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(ratios) * total_length), ratios))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        splitted_datasets = []

        i = 0
        for j in split_lengths:
            splitted_dataset = StockDataset(data_x_dir=self.data_x_dir, data_y_dir=self.data_y_dir, label_name=self.label_name)
            splitted_dataset.x_file_paths = splitted_dataset.x_file_paths[i:i+j]
            splitted_dataset.y_file_paths = splitted_dataset.y_file_paths[i:i+j]
            splitted_datasets.append(splitted_dataset)
            i += (j+mask)

        return splitted_datasets
    
    def random_split(self, ratios:List[Number]) -> List["StockDataset"]:
        """随机分割。根据给定的比例 ratios 将数据集分割成多个子数据集，且顺序随机。返回一个StockDataset类的列表。"""
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(ratios) * total_length), ratios))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        splitted_datasets = []

        base_names = [os.path.basename(file_path) for file_path in self.x_file_paths]
        random.shuffle(base_names)
        shuffled_x_file_paths = [os.path.join(self.data_x_dir, base_name) for base_name in base_names]
        shuffled_y_file_paths = [os.path.join(self.data_y_dir, base_name) for base_name in base_names]

        i = 0
        for j in split_lengths:
            splitted_dataset = StockDataset(data_x_dir=self.data_x_dir, data_y_dir=self.data_y_dir, label_name=self.label_name)
            splitted_dataset.x_file_paths = shuffled_x_file_paths[i:i+j]
            splitted_dataset.y_file_paths = shuffled_y_file_paths[i:i+j]
            splitted_datasets.append(splitted_dataset)
            i += j
            
        return splitted_datasets

class StockSequenceDataset(Dataset):
    """StockSequenceDataset 类继承自 Dataset，将 StockDataset 中的单日数据堆叠为日期序列数据"""
    def __init__(self, stock_dataset:StockDataset, seq_len:int) -> None:
        super().__init__()
        self.stock_dataset = stock_dataset # 原StockDataset对象
        self.seq_len = seq_len # 日期序列长度

    def __len__(self):
        """获取数据集长度"""
        return len(self.stock_dataset) - self.seq_len + 1
    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        """根据索引 index 从 StockDataset 中获取一个长度为 seq_len 的序列数据。"""
        X_seq = torch.stack([self.stock_dataset[i][0] for i in range(index, index+self.seq_len)], dim=0)
        y = self.stock_dataset[index+self.seq_len-1][1]
        return X_seq, y

class RandomSampleSampler(Sampler):
    def __init__(self, data_source:Dataset, num_samples_per_epoch:int):
        self.data_source:Dataset = data_source
        self.num_samples_per_epoch:int = num_samples_per_epoch

    def __iter__(self):
        indices = random.sample(range(len(self.data_source)), self.num_samples_per_epoch)
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_epoch

class RandomBatchSampler(Sampler):
    def __init__(self, data_source:Dataset, num_batches_per_epoch:int, batch_size:int):
        self.data_source:Dataset = data_source
        self.num_batches_per_epoch:int = num_batches_per_epoch
        self.batch_size:int = batch_size

    def __iter__(self):
        # 计算总的batch数量
        num_total_batches = len(self.data_source) // self.batch_size
        # 随机选择 num_batches_per_epoch 个批次的起始索引
        selected_batches = random.sample(range(num_total_batches), self.num_batches_per_epoch)
        # 生成这些批次的所有索引
        indices = []
        for batch_idx in selected_batches:
            start_idx = batch_idx * self.batch_size
            indices.extend(range(start_idx, start_idx + self.batch_size))
        return iter(indices)

    def __len__(self):
        return self.num_batches_per_epoch * self.batch_size

def parse_args():
    parser = argparse.ArgumentParser(description="Data acquisition and dataset generation.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--x_folder", type=str, required=True, help="Path of folder for x (alpha) data csv files")
    parser.add_argument("--y_folder", type=str, required=True, help="Path of folder for y (label) data csv files")
    parser.add_argument("--label_name", type=str, required=True, help="Target label name (col name in y files)")

    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.2, 0.1], help="Split ratio for train-validation-test. Default 0.7, 0.2, 0.1")
    parser.add_argument("--mask_len", type=int, default=0, help="Mask seq length to avoid model cheat(get information from future), e.g. mask 20 days when training prediction for ret20. Default 0")
    parser.add_argument("--train_seq_len", type=int, required=True, help="Sequence length (num of days) for train dataset")
    parser.add_argument("--val_seq_len", type=int, default=None, help="Sequence length (num of days) for validation dataset. If not specified, default equal to train_seq_len.")
    parser.add_argument("--test_seq_len", type=int, default=None, help="Sequence length (num of days) for test dataset. If not specified, default equal to train_seq_len.")

    parser.add_argument("--save_path", type=str, required=True, help="Path to save the dataset dictionary.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log_folder = args.log_folder
    log_name = args.log_name
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_folder, log_name)), logging.StreamHandler()])
    
    logging.debug(f"Command: {' '.join(sys.argv)}")
    logging.debug(f"Params: {vars(args)}")

    dataset = StockDataset(data_x_dir=args.x_folder,
                           data_y_dir=args.y_folder,
                           label_name=args.label_name)
    train_set, val_set, test_set = dataset.serial_split(ratios=args.split_ratio, mask=args.mask_len)
    train_set = StockSequenceDataset(train_set, seq_len=args.train_seq_len)
    val_set = StockSequenceDataset(val_set, seq_len=args.val_seq_len or args.train_seq_len)
    test_set = StockSequenceDataset(test_set, seq_len=args.test_seq_len or args.train_seq_len)

    torch.save({"train": train_set, "val": val_set, "test": test_set}, args.save_path)

# python dataset.py --x_folder "D:\PycharmProjects\SWHY\data\preprocess\alpha" --y_folder "D:\PycharmProjects\SWHY\data\preprocess\label" --label_name "ret10" --train_seq_len 20 --save_path "D:\PycharmProjects\SWHY\data\preprocess\dataset.pt"

# python dataset.py --x_folder "D:\PycharmProjects\SWHY\data\preprocess\alpha_cs_zscore" --y_folder "D:\PycharmProjects\SWHY\data\preprocess\label" --label_name "ret10" --train_seq_len 20 --save_path "D:\PycharmProjects\SWHY\data\preprocess\dataset_cs_zscore.pt"





