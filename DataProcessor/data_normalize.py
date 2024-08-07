"""
数据归一化
仅适用于机器学习的预处理，深度学习使用Norm层代替
"""

import os
import sys
import logging
import argparse
from numbers import Number
from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler 

@dataclass
class DateData:
    # 路径
    path:str
    # 日期
    date:str
    # 平均值序列
    mean_series:pd.Series
    # 标准差序列
    std_series:pd.Series
    # 最小值序列
    min_val_series:pd.Series
    # 最大值序列
    max_val_series:pd.Series
    # 中位数序列
    median_series:pd.Series
    # 平均绝对偏差序列
    mad_series:pd.Series

class DisributedNormalizer:
    """
    分布式归一化处理管线。
    用于对指定文件夹中的CSV文件进行标准化处理。标准化处理包括多种模式，如Z-score标准化、排名标准化、全局Z-score标准化、全局最小-最大标准化和全局鲁棒Z-score标准化。
    """
    def __init__(self, folder_path:str, disable_tqdm:bool=False) -> None:
        self.folder_path:str = folder_path
        self.disable_tqdm:bool = disable_tqdm

        self.date_data_list:List[DateData] = []

        self.global_mean:float
        self.global_std:float
        self.global_min:float
        self.global_max:float
        self.global_median:float
        self.global_mad:float

    def _read_and_process_file(self, file_path:str, fill_value:Number=0) -> DateData:
        # 方法读取并处理单个CSV文件，返回一个 DateData 对象，包含文件路径、日期、均值、标准差、最小值、最大值、中位数和平均绝对偏差（MAD）。
        df = pd.read_csv(file_path, index_col=0).iloc[:,2:]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(fill_value, inplace=True)
        
        date = os.path.basename(file_path).replace(".csv", "")
        mean = df.mean()
        std = df.std()
        min_val = df.min()
        max_val = df.max()
        median = df.median()
        mad = pd.Series(np.median(np.abs(df - median), axis=0))

        return DateData(path=file_path, 
                        date=date, 
                        mean_series=mean, 
                        std_series=std,
                        min_val_series=min_val,
                        max_val_series=max_val,
                        median_series=median,
                        mad_series=mad)
    
    def _load_data(self, fill_value:Number=0) -> None:
        # 加载数据。遍历指定文件夹中的所有CSV文件，调用 _read_and_process_file 方法读取和处理文件，并将结果存储在 date_data_list 中。
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        
        for file_name in tqdm(file_list, disable=self.disable_tqdm):
            file_path = os.path.join(self.folder_path, file_name)
            date_data = self._read_and_process_file(file_path, fill_value)
            self.date_data_list.append(date_data)
    
    def _calculate_global_statistics(self) -> None:
        # 计算所有数据的全局统计信息，包括均值、标准差、最小值、最大值、中位数和平均绝对偏差（MAD）。
        self.global_mean = pd.concat([datedata.mean_series for datedata in self.date_data_list], axis=1).mean().mean()
        self.global_std = pd.concat([datedata.std_series for datedata in self.date_data_list], axis=1).mean().mean()
        self.global_min = pd.concat([datedata.min_val_series for datedata in self.date_data_list], axis=1).min().min()
        self.global_max = pd.concat([datedata.max_val_series for datedata in self.date_data_list], axis=1).max().max()
        self.global_median = pd.concat([datedata.median_series for datedata in self.date_data_list], axis=1).median().mean()
        self.global_mad = pd.concat([datedata.mad_series for datedata in self.date_data_list], axis=1).median().mean()
        logging.info(f"global mean:{self.global_mean}")

    def cs_zscore(self, df:pd.DataFrame) -> pd.DataFrame:
        # 使用 StandardScaler 对数据进行截面Z-score标准化。
        cs_zscore_scaler = StandardScaler()
        return pd.DataFrame(cs_zscore_scaler.fit_transform(df), columns=df.columns)

    def cs_rank(self, df:pd.DataFrame) -> pd.DataFrame:
        # 对数据进行截面排序标准化。
        return df.rank()

    def global_zscore(self, df:pd.DataFrame) -> pd.DataFrame:
        # 计算全局均值和标准差，对数据进行全局Z-score标准化。
        return (df - self.global_mean) / self.global_std

    def global_minmax(self, df:pd.DataFrame) -> pd.DataFrame:
        # 计算全局最小值和最大值。对数据进行全局最小-最大标准化。
        return (df - self.global_min) / (self.global_max - self.global_min)

    def global_robust_zscore(self, df:pd.DataFrame) -> pd.DataFrame:
        # 计算全局中位数和平均绝对偏差（MAD），对数据进行全局鲁棒Z-score标准化。
        return (df - self.global_median) / (1.4826 * self.global_mad + 0.0001)
    
    def process(self, 
                normalization_mode:Literal["cs_zscore", "cs_rank", "global_zscore", "global_minmax", "global_robust_zscore"]="cs_zscore", 
                output_dir:Optional[str]=None):
        #选择相应的标准化方法。遍历 date_data_list，对每个文件进行标准化处理，并将结果保存到指定的输出目录中。
        logging.debug(f"MODE: {normalization_mode}")
        
        logging.debug("loading data...")
        self._load_data()

        if output_dir is None:
            output_dir = self.folder_path

        if normalization_mode == "cs_zscore":
            process_func = self.cs_zscore
        elif normalization_mode == "cs_rank":
            process_func = self.cs_rank
        elif normalization_mode == "global_zscore":
            logging.debug("calculating global statistics...")
            self._calculate_global_statistics()
            process_func = self.global_zscore
        elif normalization_mode == "global_minmax":
            logging.debug("calculating global statistics...")
            self._calculate_global_statistics()
            process_func = self.global_minmax
        elif normalization_mode == "global_robust_zscore":
            logging.debug("calculating global statistics...")
            self._calculate_global_statistics()
            process_func = self.global_robust_zscore
        else:
            raise NotImplementedError()

        logging.debug("doing normalization for each file...")
        for date_data in tqdm(self.date_data_list, disable=self.disable_tqdm):
            output_path = os.path.join(output_dir, f"{date_data.date}.csv")
            df = pd.read_csv(date_data.path, index_col=0)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            df_index = df.iloc[:,:2]
            df_value = df.iloc[:,2:]
            normalized_df = process_func(df=df_value)
            normalized_df = pd.concat([df_index, normalized_df], axis=1)
            normalized_df.to_csv(output_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Data Normalizer.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--data_folder", type=str, required=True, help="Path of folder for csv files")
    parser.add_argument("--save_folder", type=str, default=None, help="Path of folder for Normalizer to save processed result. If not specified, files in data folder will be replaced.")
    parser.add_argument("--mode", type=str, default="cs_zscore", help="Normalization mode, literally `cs_zscore`, `cs_rank`, `global_zscore`, `global_minmax` or `global_robust_zscore`. Default `cs_score`.")

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
    
    save_folder = args.save_folder
    if save_folder is None:
        save_folder = args.data_folder
    if not os.path.exists(save_folder):    
        os.makedirs(save_folder)

    logging.debug(f"Command: {' '.join(sys.argv)}")
    logging.debug(f"Params: {vars(args)}")

    normalizer = DisributedNormalizer(folder_path=args.data_folder)
    normalizer.process(normalization_mode=args.mode, output_dir=save_folder)

    logging.debug(f"{args.mode} normalization complete.")

# python data_normalize.py --log_folder "log" --data_folder "data\preprocess\alpha" --mode "cs_zscore" --save_folder "data\preprocess\alpha_cs_zscore"
# python data_normalize.py --log_folder "log" --data_folder "data\preprocess\alpha" --mode "global_zscore" --save_folder "data\preprocess\alpha_global_zscore"


