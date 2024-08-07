"""
数据丢弃
暂时不用。因为全0行太多，若严格丢弃则几乎不剩数据。
"""

import os
import logging
import argparse
from typing import List

import pandas as pd
from tqdm import tqdm


class DropProcessor:
    """
    全零行丢弃处理管线。
    用于处理包含特定列全为零的行（即“零行”）的 CSV 文件。
    """
    def __init__(self, data_folder:str, save_folder:str, col_names:List[str]):
        self.data_folder:str = data_folder
        self.save_folder:str = save_folder or data_folder

        self.col_names:str = col_names
        self.all_zero_row_indices:str = set()

    def find_zero_rows(self, df):
        # 查找零行
        zero_rows = df[(df[self.col_names] == 0).all(axis=1)]
        return zero_rows.index

    def record_zero_rows(self, zero_row_indices):
        # 记录零行索引
        self.all_zero_row_indices.update(zero_row_indices)

    def remove_zero_rows(self, df:pd.DataFrame):
        # 移除零行
        df = df.drop(self.all_zero_row_indices.intersection(df.index)).reset_index(drop=True)
        return df

    def check_csv(self, file_path):
        # 检查 CSV 文件
        df = pd.read_csv(file_path, index_col=[0])
        zero_row_indices = self.find_zero_rows(df)
        self.record_zero_rows(zero_row_indices)

    def apply_removals(self, file_path, save_path):
        # 应用移除操作
        df = pd.read_csv(file_path, index_col=[0])
        df = self.remove_zero_rows(df)
        df.to_csv(save_path)

    def check(self):
        # 检查所有 CSV 文件
        for filename in tqdm(os.listdir(self.data_folder)):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_folder, filename)
                self.check_csv(file_path)
        
    def process(self):
        # 处理所有 CSV 文件
        for filename in tqdm(os.listdir(self.save_folder)):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_folder, filename)
                save_path = os.path.join(self.save_folder, filename)
                self.apply_removals(file_path, save_path)
            
def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessor. Usage: data.py [-h] [--log_folder LOG_FOLDER] [--log_name LOG_NAME] {alpha,label} --data_folder DATA_FOLDER --save_folder SAVE_FOLDER [--fill_value FILL_VALUE --merge_mode MERGE_MODE]")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="name of log file. Default `log.txt`")

    parser.add_argument("--data_folder", type=str, required=True, help="Path of folder for data csv files (contains subdir `alpha` and `label`)")
    parser.add_argument("--save_folder", type=str, default=None, help="Path of folder for data csv files (contains subdir `alpha` and `label`). If None specified, will use data_folder i.e. process inplace. Default None")
    parser.add_argument("--alpha_col_names", type=str, nargs="+", default=[f'Alpha_{str(i).zfill(3)}' for i in range(1, 6)], help="Alpha column name, whose zero value that droping row based on")
    parser.add_argument("--label_col_names", type=str, nargs="+", default=["ret5", "ret10", "ret20", "spret5", "spret10", "spret20"], help="Label column name, whose zero value that droping row based on")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    log_folder = args.log_folder
    log_name = args.log_name
    
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "alpha"), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "label"), exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_folder, log_name)), logging.StreamHandler()])
    
    logging.info("Drop Process Start")
    
    alpha_processor = DropProcessor(data_folder=os.path.join(args.data_folder, "alpha"),
                                    save_folder=os.path.join(args.save_folder, "alpha"),
                                    col_names=args.alpha_col_names)
    label_processor = DropProcessor(data_folder=os.path.join(args.data_folder, "label"),
                                    save_folder=os.path.join(args.save_folder, "label"),
                                    col_names=args.label_col_names)

    alpha_processor.check()
    label_processor.check()
    logging.info(f"{len(alpha_processor.all_zero_row_indices)} rows will be dropped in Alpha")
    logging.info(f"{len(label_processor.all_zero_row_indices)} rows will be dropped in Label")
    
    alpha_processor.process()
    label_processor.process()
    logging.info("Drop Process Complete")

# python data_drop.py --data_folder "data\preprocess" --save_folder "data\dropped" --log_folder "log"