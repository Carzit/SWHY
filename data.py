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

@dataclass
class FileData:
    path: str
    dataframe: pd.DataFrame

@dataclass
class AlphaData(FileData):
    codes: List[str]
    dates: List[str]
    factor: str

@dataclass
class LabelData(FileData):
    codes: List[str]
    dates: List[str]
    label: str


class AlphaProcessor:
    def __init__(self, folder_path:str, disable_tqdm:bool=False):
        self.folder_path:str = folder_path
        self.disable_tqdm:bool = disable_tqdm

        self.common_dates:List = None
        self.common_codes:List = None
        self.factors:List = []

        self.alpha_data_list:List[AlphaData] = []

    def _read_and_process_file(self, file_path, fill_value=0) -> AlphaData:
        # read pickle files and process missing values
        df = pd.read_pickle(file_path)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(fill_value, inplace=True)

        codes = df.columns.tolist()
        dates = df.index.tolist()
        factor = os.path.basename(file_path).replace("OHO_", "").replace(".pkl", "")
        self.factors.append(factor)

        return AlphaData(path=file_path, dataframe=df, codes=codes, dates=dates, factor=factor)

    def _load_data(self, fill_value=0) -> None:
        # load all data in specified file folder
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
    
    def _common_filter(self) -> None:
        # Filter dataframes to keep only common codes and dates
        for alpha_data in tqdm(self.alpha_data_list, disable=self.disable_tqdm):
            alpha_data.dataframe = alpha_data.dataframe.loc[self.common_dates, self.common_codes]
            alpha_data.codes = self.common_codes
            alpha_data.dates = self.common_dates
    
    def _merge_stock_code(self, stock_code:str) -> pd.DataFrame:
        # merge data and save according to different stock code
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
        # merge data and save according to different date
        merged_df = pd.DataFrame(index=self.common_codes, columns=[])
        merged_df.index.name = 'stock_code'

        for alpha_data in self.alpha_data_list:
            if date in alpha_data.dates:
                series = alpha_data.dataframe.loc[date]
                series.name = alpha_data.factor
                merged_df = merged_df.join(series, how='left')
                i+=1

        merged_df.insert(0, 'date', date)
        merged_df.reset_index(inplace=True)
        return merged_df

    def _merge_all(self) -> pd.DataFrame:
        # merge all into a large dataframe
        # [WARNING] out of memery  

        # Initialize merged dataframe with common codes and dates
        merged_df = pd.DataFrame(index=pd.MultiIndex.from_product([self.common_codes, self.common_dates], names=["stock_code", "date"]))
        
        
        for alpha_data in tqdm(self.alpha_data_list, disable=self.disable_tqdm):
            # Reshape the dataframe to have 'code' and 'date' as index
            reshaped_df = alpha_data.dataframe.stack().reset_index()
            reshaped_df.columns = ['date', 'stock_code', alpha_data.factor]
            reshaped_df.set_index(['stock_code', 'date'], inplace=True)
            merged_df = merged_df.join(reshaped_df, how='left')
        
        return merged_df

    def process(self, 
                fill_value:Number=0, 
                skip_filter:bool=True, 
                merge_mode:Literal["date", "stock_code", "all"]="date", 
                save_folder:Optional[str]=os.curdir) -> None:
        logging.debug("loading data...")
        self._load_data(fill_value)

        if not skip_filter:
            logging.debug("doing common filter...")
            self._common_filter()

        logging.debug("recording extra info")
        with open(os.path.join(save_folder, "alpha_extra_info.json"), "w") as f:
            json.dump({"dates": self.common_dates,
                       "stock_codes": self.common_codes}, f)
        
        logging.debug("merging data...")
        match merge_mode:
            case "date":
                for date in tqdm(self.common_dates, disable=self.disable_tqdm):
                    merged_df = self._merge_date(date)
                    merged_df.to_csv(os.path.join(save_folder, f"{date}.csv"))
            case "stock_code":
                for code in tqdm(self.common_codes, disable=self.disable_tqdm):
                    merged_df = self._merge_stock_code(code)
                    merged_df.to_csv(os.path.join(save_folder, f"{code}.csv"))
            case "all":
                merged_df = self._merge_all()
                merged_df.to_csv(os.path.join(save_folder, f"merged_alpha_data.csv"))

class LabelProcessor:
    def __init__(self, folder_path:str, disable_tqdm:bool=False):
        self.folder_path:str = folder_path
        self.disable_tqdm:bool = disable_tqdm

        self.common_dates:List = None
        self.common_codes:List = None
        self.labels:List = []

        self.label_data_list:List[LabelData] = []

    def _read_and_process_file(self, file_path:str, fill_value:Number=0) -> LabelData:
        # read pickle files and process missing values
        df = pd.read_pickle(file_path)

        df.replace([np.inf, -np.inf], np.nan)
        df.fillna(fill_value, inplace=True)

        codes = df.columns.tolist()
        dates = df.index.tolist()
        label = os.path.basename(file_path).replace("label_", "").replace(".pkl", "")
        self.labels.append(label)

        return LabelData(path=file_path, dataframe=df, codes=codes, dates=dates, label=label)

    def _load_data(self, fill_value=0) -> None:
        # load all data in specified file folder
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
    
    def _common_filter(self) -> None:
        # Filter dataframes to keep only common codes and dates
        for label_data in tqdm(self.label_data_list, disable=self.disable_tqdm):
            label_data.dataframe = label_data.dataframe.loc[self.common_dates, self.common_codes]
            label_data.codes = self.common_codes
            label_data.dates = self.common_dates
    
    def _merge_stock_code(self, stock_code:str) -> pd.DataFrame:
        # merge data and save according to different stock code
        merged_df = pd.DataFrame(index=self.common_dates, columns=[])
        merged_df.index.name = 'date'

        for label_data in tqdm(self.label_data_list, disable=self.disable_tqdm):
            if stock_code in label_data.codes:
                series = label_data.dataframe[stock_code]
                series.name = label_data.label
                merged_df = merged_df.join(series, how='left')

        merged_df.insert(0, 'stock_code', stock_code)
        merged_df.reset_index(inplace=True)
        return merged_df
    
    def _merge_date(self, date:str) -> pd.DataFrame:
        # merge data and save according to different date
        merged_df = pd.DataFrame(index=self.common_codes, columns=[])
        merged_df.index.name = 'stock_code'

        for label_data in tqdm(self.label_data_list, disable=self.disable_tqdm):
            if date in label_data.dates:
                series = label_data.dataframe.loc[date]
                series.name = label_data.label
                merged_df = merged_df.join(series, how='left')

        merged_df.insert(0, 'date', date)
        merged_df.reset_index(inplace=True)
        return merged_df

    def _merge_all(self) -> pd.DataFrame:
        # merge all into a large dataframe
        # [WARNING] out of memery  

        # Initialize merged dataframe with common codes and dates
        merged_df = pd.DataFrame(index=pd.MultiIndex.from_product([self.common_codes, self.common_dates], names=["stock_code", "date"]))
        
        
        for label_data in tqdm(self.label_data_list, disable=self.disable_tqdm):
            # Reshape the dataframe to have 'code' and 'date' as index
            reshaped_df = label_data.dataframe.stack().reset_index()
            reshaped_df.columns = ['date', 'stock_code', label_data.label]
            reshaped_df.set_index(['stock_code', 'date'], inplace=True)
            merged_df = merged_df.join(reshaped_df, how='left')
        
        return merged_df.reset_index()

    def process(self, 
                fill_value:Number=0, 
                skip_filter:bool=True, 
                merge_mode:Literal["date", "stock_code", "all"]="date", 
                save_folder:Optional[str]=os.curdir) -> None:
        logging.debug(f"MODE: {merge_mode}")

        logging.debug("loading data...")
        self._load_data(fill_value)

        if not skip_filter:
            logging.debug("doing common filter...")
            self._common_filter()

        logging.debug("recording extra info")
        with open(os.path.join(save_folder, "label_extra_info.json"), "w") as f:
            json.dump({"dates": self.common_dates,
                       "stock_codes": self.common_codes}, f)
            
        
        logging.debug("merging data...")
        match merge_mode:
            case "date":
                for date in tqdm(self.common_dates, disable=self.disable_tqdm):
                    merged_df = self._merge_date(date)
                    merged_df.to_csv(os.path.join(save_folder, f"{date}.csv"))
            case "stock_code":
                for code in tqdm(self.common_codes, disable=self.disable_tqdm):
                    merged_df = self._merge_stock_code(code)
                    merged_df.to_csv(os.path.join(save_folder, f"{code}.csv"))
            case "all":
                merged_df = self._merge_all()
                merged_df.to_csv(os.path.join(save_folder, f"merged_label_data.csv"))


def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessor. Usage: data.py [-h] [--log_folder LOG_FOLDER] [--log_name LOG_NAME] {alpha,label} --data_folder DATA_FOLDER --save_folder SAVE_FOLDER [--fill_value FILL_VALUE --merge_mode MERGE_MODE]")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="name of log file. Default `log.txt`")

    subparsers = parser.add_subparsers(dest='processor', help='Data Processor, `alpha` or `label`')

    parser_alpha = subparsers.add_parser('alpha', help='Processor for alpha data')
    parser_alpha.add_argument("--data_folder", type=str, required=True, help="Path of folder for alpha pickle files")
    parser_alpha.add_argument("--save_folder", type=str, required=True, help="Path of folder for AlphaProcessor to save processed result")
    parser_alpha.add_argument("--fill_value", type=float, default=0, help="Filling value for missing value in AlphaProcessor")
    parser_alpha.add_argument("--merge_mode", type=str, default="date", help="Merge mode for alpha data, `date`, `stock_code` or `all`. Default `date`")

    parser_label = subparsers.add_parser('label', help='Processor for label data')
    parser_label.add_argument("--data_folder", type=str, required=True, help="path of folder for label pickle files")
    parser_label.add_argument("--save_folder", type=str, required=True, help="path of folder for LabelProcessor to save processed result")
    parser_label.add_argument("--fill_value", type=float, default=0, help="fill value for missing value in LabelProcessor")
    parser_label.add_argument("--merge_mode", type=str, default="date", help="Merge mode for label data, `date`, `stock_code` or `all`. Default `all`")

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
    
    if not os.path.exists(args.save_folder):    
        os.makedirs(args.save_folder)

    logging.debug(f"Command: {' '.join(sys.argv)}")
    logging.debug(f"Params: {vars(args)}")
    
    match args.processor:
        case "alpha":
            processor = AlphaProcessor(args.data_folder)
            processor.process(fill_value=args.fill_value, merge_mode=args.merge_mode, save_folder=args.save_folder)
        case "label":
            processor = LabelProcessor(args.data_folder)
            processor.process(fill_value=args.fill_value, merge_mode=args.merge_mode, save_folder=args.save_folder)

# Example
# python data.py --log_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\preprocess\log" alpha --data_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\new\Alpha101" --save_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\preprocess\alpha"
# python data.py --log_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\preprocess\log"  label  --data_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\new\label" --save_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\preprocess\label"    