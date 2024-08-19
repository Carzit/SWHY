"""
数据构建
对原Alpha和Label的pkl文件进行缺失值处理、数据对齐和重新组织
"""

import os
import sys
import logging
import argparse
from numbers import Number
from typing import List, Literal, Optional
from dataclasses import dataclass

import json
import numpy as np
import pandas as pd
from tqdm import tqdm


# 定义一个FileData类，包含path和dataframe两个属性。作为基类，描述每个数据文件。
@dataclass
class FileData:
    path: str
    dataframe: pd.DataFrame

# 定义一个AlphaData类，继承自FileData类，包含codes、dates和factor三个属性。用于描述每个Alpha文件
@dataclass
class AlphaData(FileData):
    codes: List[str]
    dates: List[str]
    factor: str

# 定义一个LabelData类，继承自FileData类，包含codes、dates和label三个属性。用于描述每个Label文件
@dataclass
class LabelData(FileData):
    codes: List[str]
    dates: List[str]
    label: str

def save_dataframe(df:pd.DataFrame, path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl"):
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

def load_dataframe(path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl"):
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

class AlphaProcessor:
    """
    Alpha处理管线
    将以因子名为文件名、以股票代码为列名、以日期为行名的Alpha数据重新组织
    """
    def __init__(self, folder_path:str, disable_tqdm:bool=False):
        self.folder_path:str = folder_path #待处理数据所在文件夹的路径
        self.disable_tqdm:bool = disable_tqdm #是否禁用进度条

        self.common_dates:List = None #共有日期（用于后续做数据对齐）
        self.common_codes:List = None #共有股票代码（用于后续做数据对齐）
        self.factors:List = [] #因子名

        self.alpha_data_list:List[AlphaData] = [] #所有Alpha数据

    def _read_and_process_file(self, file_path, fill_value=0) -> AlphaData:
        # 读取指定路径的pickle文件，处理缺失值和无限值，并返回一个 AlphaData 对象。
        df:pd.DataFrame = pd.read_pickle(file_path)

        #df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(fill_value, inplace=True)

        codes = df.columns.tolist()
        dates = df.index.tolist()
        factor = os.path.basename(file_path).replace("OHO_", "").replace(".pkl", "")
        self.factors.append(factor)

        return AlphaData(path=file_path, dataframe=df, codes=codes, dates=dates, factor=factor)
    
    def load_data(self, fill_value=0) -> None:
        # 读取指定文件夹中的所有pkl文件，处理数据并存储在 alpha_data_list 中，同时计算所有文件中共同的日期和股票代码。
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
        
        for file_name in tqdm(file_list, disable=self.disable_tqdm):
            file_path = os.path.join(self.folder_path, file_name)
            alpha_data = self._read_and_process_file(file_path, fill_value)
            
            if self.common_dates is None:
                self.common_dates = pd.Index(alpha_data.dates)
            else:
                self.common_dates = self.common_dates.intersection(alpha_data.dates)

            if self.common_codes is None:
                self.common_codes = pd.Index(alpha_data.codes)
            else:
                self.common_codes = self.common_codes.intersection(alpha_data.codes)

            self.alpha_data_list.append(alpha_data)

        self.common_dates = list(self.common_dates)
        self.common_codes = list(self.common_codes)
    
    def common_filter(self) -> None:
        # 过滤数据，只保留所有文件中共同的日期和股票代码。
        for alpha_data in tqdm(self.alpha_data_list, disable=self.disable_tqdm):
            alpha_data.dataframe = alpha_data.dataframe.loc[self.common_dates, self.common_codes]
            alpha_data.codes = self.common_codes
            alpha_data.dates = self.common_dates
    
    def _merge_stock_code(self, stock_code:str) -> pd.DataFrame:
        # 根据不同的股票代码合并数据，并返回一个以日期为行名，以因子为列名的DataFrame。
        merged_df = pd.DataFrame(index=self.common_dates, columns=[])
        merged_df.index.name = 'date'

        for alpha_data in self.alpha_data_list:
            if stock_code in alpha_data.codes:
                series = alpha_data.dataframe[stock_code]
                series.name = alpha_data.factor
                merged_df = merged_df.join(series, how='left')

        merged_df.insert(0, 'stock_code', stock_code)
        merged_df.reset_index(inplace=True)
        return merged_df
    
    def _merge_date(self, date:str) -> pd.DataFrame:
        # 根据不同的日期合并数据，并返回一个以股票代码为行名，以因子为列名的DataFrame。
        merged_df = pd.DataFrame(index=self.common_codes, columns=[])
        merged_df.index.name = 'stock_code'

        for alpha_data in self.alpha_data_list:
            if date in alpha_data.dates:
                series = alpha_data.dataframe.loc[date]
                series.name = alpha_data.factor
                merged_df = merged_df.join(series, how='left')

        merged_df.insert(0, 'date', date)
        merged_df.reset_index(inplace=True)
        return merged_df

    def _merge_all(self) -> pd.DataFrame:
        # 将所有数据合并成一个大的DataFrame，以日期为第一列index，股票代码为第二列index，以因子为列名。
        merged_df = pd.DataFrame(index=pd.MultiIndex.from_product([self.common_codes, self.common_dates], names=["stock_code", "date"]))
        
        for alpha_data in tqdm(self.alpha_data_list, disable=self.disable_tqdm):
            reshaped_df = alpha_data.dataframe.stack().reset_index()
            reshaped_df.columns = ['date', 'stock_code', alpha_data.factor]
            reshaped_df.set_index(['stock_code', 'date'], inplace=True)
            merged_df = merged_df.join(reshaped_df, how='left')
        
        return merged_df

    def process(self, 
                merge_mode:Literal["date", "stock_code", "all"]="date", 
                save_folder:Optional[str]=os.curdir,
                save_format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> None:
        # 根据指定的合并模式（日期、股票代码或全部）处理数据，并将结果以csv文件格式，保存到指定文件夹中。
        logging.debug("recording extra info")
        with open(os.path.join(save_folder, "alpha_extra_info.json"), "w") as f:
            json.dump({"dates": self.common_dates,
                       "stock_codes": self.common_codes}, f)
        
        logging.debug("merging data...")
        if merge_mode == "date":
            for date in tqdm(self.common_dates, disable=self.disable_tqdm):
                merged_df = self._merge_date(date)
                save_dataframe(df=merged_df,
                               path=os.path.join(save_folder, f"{date}.{save_format}"),
                               format=save_format)
        elif merge_mode == "stock_code":
            for code in tqdm(self.common_codes, disable=self.disable_tqdm):
                merged_df = self._merge_stock_code(code)
                save_dataframe(df=merged_df,
                               path=os.path.join(save_folder, f"{code}.{save_format}"),
                               format=save_format)
        elif merge_mode == "all":
            merged_df = self._merge_all()
            save_dataframe(df=merged_df,
                           path=os.path.join(save_folder, f"all.{save_format}"),
                           format=save_format)

class LabelProcessor:
    """
    Label处理管线
    将以因子名为文件名、以股票代码为列名、以日期为行名的Alpha数据重新组织
    """
    def __init__(self, folder_path:str, disable_tqdm:bool=False):
        self.folder_path:str = folder_path #待处理数据所在文件夹的路径
        self.disable_tqdm:bool = disable_tqdm #是否禁用进度条

        self.common_dates:List = None #共有日期（用于后续做数据对齐）
        self.common_codes:List = None #共有股票代码（用于后续做数据对齐）
        self.labels:List = [] #标签名

        self.label_data_list:List[LabelData] = [] #所有标签数据

    def _read_and_process_file(self, file_path:str, fill_value:Number=0) -> LabelData:
        # 读取指定路径的pickle文件，处理缺失值和无限值，并返回一个 LabelData 对象。
        df:pd.DataFrame = pd.read_pickle(file_path)

        #df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(fill_value, inplace=True)

        codes = df.columns.tolist()
        dates = df.index.tolist()
        label = os.path.basename(file_path).replace("label_", "").replace(".pkl", "")
        self.labels.append(label)

        return LabelData(path=file_path, dataframe=df, codes=codes, dates=dates, label=label)

    def load_data(self, fill_value=0) -> None:
        # 读取指定文件夹中的所有pkl文件，处理数据并存储在 label_data_list 中，同时计算所有文件中共同的日期和股票代码。
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
        
        for file_name in tqdm(file_list, disable=self.disable_tqdm):
            file_path = os.path.join(self.folder_path, file_name)
            label_data = self._read_and_process_file(file_path, fill_value)
            
            if self.common_dates is None:
                self.common_dates = pd.Index(label_data.dates)
            else:
                self.common_dates = self.common_dates.intersection(label_data.dates)

            if self.common_codes is None:
                self.common_codes = pd.Index(label_data.codes)
            else:
                self.common_codes = self.common_codes.intersection(label_data.codes)

            self.label_data_list.append(label_data)

        self.common_dates = list(self.common_dates)
        self.common_codes = list(self.common_codes)
    
    def common_filter(self) -> None:
        # 过滤数据，只保留所有文件中共同的日期和股票代码。
        for label_data in tqdm(self.label_data_list, disable=self.disable_tqdm):
            label_data.dataframe = label_data.dataframe.loc[self.common_dates, self.common_codes]
            label_data.codes = self.common_codes
            label_data.dates = self.common_dates
    
    def _merge_stock_code(self, stock_code:str) -> pd.DataFrame:
        # 根据不同的股票代码合并数据，并返回一个以日期为行名，以标签为列名的DataFrame。
        merged_df = pd.DataFrame(index=self.common_dates, columns=[])
        merged_df.index.name = 'date'

        for label_data in self.label_data_list:
            if stock_code in label_data.codes:
                series = label_data.dataframe[stock_code]
                series.name = label_data.label
                merged_df = merged_df.join(series, how='left')

        merged_df.insert(0, 'stock_code', stock_code)
        merged_df.reset_index(inplace=True)
        return merged_df
    
    def _merge_date(self, date:str) -> pd.DataFrame:
        # 根据不同的日期合并数据，并返回一个以股票代码为行名，以标签为列名的DataFrame。
        merged_df = pd.DataFrame(index=self.common_codes, columns=[])
        merged_df.index.name = 'stock_code'

        for label_data in self.label_data_list:
            if date in label_data.dates:
                series = label_data.dataframe.loc[date]
                series.name = label_data.label
                merged_df = merged_df.join(series, how='left')

        merged_df.insert(0, 'date', date)
        merged_df.reset_index(inplace=True)
        return merged_df

    def _merge_all(self) -> pd.DataFrame:
        # 将所有数据合并成一个大的DataFrame，以日期为第一列index，股票代码为第二列index，以标签为列名。
        merged_df = pd.DataFrame(index=pd.MultiIndex.from_product([self.common_codes, self.common_dates], names=["stock_code", "date"])) 
        
        for label_data in tqdm(self.label_data_list, disable=self.disable_tqdm):
            reshaped_df = label_data.dataframe.stack().reset_index()
            reshaped_df.columns = ['date', 'stock_code', label_data.label]
            reshaped_df.set_index(['stock_code', 'date'], inplace=True)
            merged_df = merged_df.join(reshaped_df, how='left')
        
        return merged_df.reset_index()

    def process(self, 
                merge_mode:Literal["date", "stock_code", "all"]="date", 
                save_folder:Optional[str]=os.curdir,
                save_format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> None:
        # 根据指定的合并模式（日期、股票代码或全部）处理数据，并将结果以csv文件格式，保存到指定文件夹中。
        logging.debug(f"MODE: {merge_mode}")

        logging.debug("recording extra info")
        with open(os.path.join(save_folder, "label_extra_info.json"), "w") as f:
            json.dump({"dates": self.common_dates,
                       "stock_codes": self.common_codes}, f)
        
        logging.debug("merging data...")
        if merge_mode == "date":
            for date in tqdm(self.common_dates, disable=self.disable_tqdm):
                merged_df = self._merge_date(date)
                save_dataframe(df=merged_df,
                               path=os.path.join(save_folder, f"{date}.{save_format}"),
                               format=save_format)
        elif merge_mode == "stock_code":
            for code in tqdm(self.common_codes, disable=self.disable_tqdm):
                merged_df = self._merge_stock_code(code)
                save_dataframe(df=merged_df,
                               path=os.path.join(save_folder, f"{code}.{save_format}"),
                               format=save_format)
        elif merge_mode == "all":
            merged_df = self._merge_all()
            save_dataframe(df=merged_df,
                           path=os.path.join(save_folder, f"all.{save_format}"),
                           format=save_format)

class DataAligner:
    """
    数据对齐处理管线
    将Alpha数据和Label数据进行对齐
    """

    def __init__(self, alpha_processor:AlphaProcessor, label_processor:LabelProcessor) -> None: # 初始化DataAligner类，传入alpha_processor和label_processor的引用。
        self.alpha_processor:AlphaProcessor = alpha_processor
        self.label_processor:LabelProcessor = label_processor

    def align(self, lst1:List, lst2:List) -> List:
        # 对传入的两个列表进行对齐操作，返回两个列表的交集。
        set1 = set(lst1)
        set2 = set(lst2)
        common_set = set1.intersection(set2)
        return list(common_set)
        
    def process(self) -> None:
        # 对alpha_processor和label_processor进行对齐操作
        alpha_dates = self.alpha_processor.common_dates
        alpha_codes = self.alpha_processor.common_codes
        label_dates = self.label_processor.common_dates
        label_codes = self.label_processor.common_codes
        
        common_dates = self.align(alpha_dates, label_dates)
        common_codes = self.align(alpha_codes, label_codes)
        
        self.alpha_processor.common_dates = common_dates
        self.label_processor.common_dates = common_dates
        self.alpha_processor.common_codes = common_codes
        self.label_processor.common_codes = common_codes

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessor. Usage: data.py [-h] [--log_folder LOG_FOLDER] [--log_name LOG_NAME] {alpha,label} --data_folder DATA_FOLDER --save_folder SAVE_FOLDER [--fill_value FILL_VALUE --merge_mode MERGE_MODE]")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--alpha_folder", type=str, required=True, help="Path of folder for alpha pickle files")
    parser.add_argument("--label_folder", type=str, required=True, help="Path of folder for label pickle files")
    parser.add_argument("--save_folder", type=str, required=True, help="Path of folder for Processor to save processed result in subdir `alpha` and `label`")
    parser.add_argument("--save_format", type=str, default="pkl", help="File format to save, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`")
    parser.add_argument("--fill_value", type=float, default=0, help="Filling value for missing value in AlphaProcessor")
    parser.add_argument("--merge_mode", type=str, default="date", help="Merge mode for alpha data, `date`, `stock_code` or `all`. Default `date`")

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
    if not os.path.exists(args.save_folder):    
        os.makedirs(args.save_folder)
        os.makedirs(os.path.join(args.save_folder, "alpha"))
        os.makedirs(os.path.join(args.save_folder, "label"))

    alpha_processor = AlphaProcessor(args.alpha_folder)
    label_processor = LabelProcessor(args.label_folder)

    logging.debug("Loading Alpha data...")
    alpha_processor.load_data()

    logging.debug("Loading Label data...")
    label_processor.load_data()

    logging.debug("Doing data alignment...")
    aligner = DataAligner(alpha_processor=alpha_processor, label_processor=label_processor)
    aligner.process()
    alpha_processor.common_filter()
    label_processor.common_filter()

    logging.debug("Doing Alpha data split...")
    alpha_processor.process(merge_mode=args.merge_mode, save_folder=os.path.join(args.save_folder, "alpha"), save_format=args.save_format)
    logging.debug(f"Alpha date data saved to {os.path.join(args.save_folder, "alpha")}")
    
    logging.debug("Doing Label data split...")
    label_processor.process(merge_mode=args.merge_mode, save_folder=os.path.join(args.save_folder, "label"), save_format=args.save_format)
    logging.debug(f"Label date data saved to {os.path.join(args.save_folder, "label")}")
    
    logging.debug("Data Construct Accomplished")

# python data_construct.py --alpha_folder "data\Alpha101" --label_folder "data\label" --save_folder "data\preprocess" --merge_mode "date" --log_folder "log"