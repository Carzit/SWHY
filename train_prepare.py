import os
import sys
import json
import random
import shutil
import logging
import argparse
from typing import List, Tuple, Literal, Callable, Optional

from tqdm import tqdm


class DataPoolProcessor:
    def __init__(self, 
                 source_x_dir: str, 
                 source_y_dir: str, 
                 dest_dir: str, 
                 train_pool_count: Optional[int] = None, 
                 test_pool_count: Optional[int] = None, 
                 max_files_per_pool: Optional[int] = None, 
                 train_test_ratio: float = 0.8, 
                 split_mode: Literal["random", "serial"] = "random",
                 shutil_mode: Literal["move", "copy"] = "move"):
        self.source_x_dir = source_x_dir
        self.source_y_dir = source_y_dir
        self.dest_dir = dest_dir

        self.train_pool_count = train_pool_count
        self.test_pool_count = test_pool_count
        self.max_files_per_pool = max_files_per_pool
        self.train_test_ratio = train_test_ratio

        self.split_mode = split_mode
        self.shutil_mode = shutil_mode

        self.file_structure = {"train_data": {"X": {}, "Y": {}}, "test_data": {"X": {}, "Y": {}}}

    def get_file_list(self, source_dir: str) -> List[str]:
        files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.csv')])
        return files

    def split_data(self, files: List[str]) -> Tuple[List[str], List[str]]:
        train_size = int(len(files) * self.train_test_ratio)
        train_files = files[:train_size]
        test_files = files[train_size:]
        return train_files, test_files

    def distribute_to_pools(self, files: List[str], pool_count: int, dest_subdir: str) -> None:      
        match self.split_mode:
            case "random":
                random.shuffle(files)
            case "serial":
                pass

        match self.shutil_mode:
            case "move":
                shutil_ops = shutil.move
            case "copy":
                shutil_ops = shutil.copy
        
        pool_size = self.max_files_per_pool if self.max_files_per_pool else len(files) // pool_count
        for i in tqdm(range(0, len(files), pool_size)):
            pool_files = files[i:i + pool_size]
            pool_index = i // pool_size + 1
            pool_dir = os.path.join(self.dest_dir, dest_subdir, f"pool_{pool_index}")
            os.makedirs(pool_dir, exist_ok=True)
            self.file_structure[dest_subdir.split("/")[0]][dest_subdir.split("/")[1]][f"pool_{pool_index}"] = []
            for file in pool_files:
                shutil_ops(file, pool_dir)
                self.file_structure[dest_subdir.split("/")[0]][dest_subdir.split("/")[1]][f"pool_{pool_index}"].append(os.path.basename(file))

    def save_file_structure(self) -> None:
        with open(os.path.join(self.dest_dir, 'file_structure.json'), 'w') as f:
            json.dump(self.file_structure, f, indent=4)

    def calculate_pool_sizes(self, pool_dir:str) -> float:
        pool_sizes = []
        for pool in os.listdir(pool_dir):
            pool_path = os.path.join(pool_dir, pool)
            total_size = sum(os.path.getsize(os.path.join(pool_path, f)) for f in os.listdir(pool_path))
            pool_sizes.append(total_size / (1024 * 1024))  # Convert to MB
        return pool_sizes

    def process(self) -> None:
        x_files = self.get_file_list(self.source_x_dir)
        y_files = self.get_file_list(self.source_y_dir)

        # Ensure x_files and y_files are matched
        assert len(x_files) == len(y_files)
        for xf, yf in zip(x_files, y_files):
            assert os.path.basename(xf) == os.path.basename(yf)

        # Split the data into training and testing sets
        train_x_files, test_x_files = self.split_data(x_files)
        train_y_files, test_y_files = self.split_data(y_files)
        
        logging.debug(f"Total files: {len(x_files)}")
        logging.debug(f"Training files: {len(train_x_files)}")
        logging.debug(f"Testing files: {len(test_x_files)}")

        # Determine pool counts if not specified
        if not self.train_pool_count:
            self.train_pool_count = len(train_x_files) // self.max_files_per_pool
        if not self.test_pool_count:
            self.test_pool_count = len(test_x_files) // self.max_files_per_pool

        # Distribute files into pools
        self.distribute_to_pools(train_x_files, self.train_pool_count, "train_data/X")
        self.distribute_to_pools(train_y_files, self.train_pool_count, "train_data/Y")
        self.distribute_to_pools(test_x_files, self.test_pool_count, "test_data/X")
        self.distribute_to_pools(test_y_files, self.test_pool_count, "test_data/Y")

        # Save file structure to a JSON file
        self.save_file_structure()

        # Calculate and print average pool sizes for training data
        train_x_pool_sizes = self.calculate_pool_sizes(os.path.join(self.dest_dir, 'train_data', 'X'))
        train_y_pool_sizes = self.calculate_pool_sizes(os.path.join(self.dest_dir, 'train_data', 'Y'))
        avg_train_x_pool_size = sum(train_x_pool_sizes) / len(train_x_pool_sizes)
        avg_train_y_pool_size = sum(train_y_pool_sizes) / len(train_y_pool_sizes)
        
        logging.debug(f"Average train_x pool size: {avg_train_x_pool_size:.2f} MB")
        logging.debug(f"Average train_y pool size: {avg_train_y_pool_size:.2f} MB")

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Data Normalizer.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--source_x_dir", type=str, required=True, help="Path of folder for train X files (Alpha Factors)")
    parser.add_argument("--source_y_dir", type=str, required=True, help="Path of folder for train Y files (Labels)")
    parser.add_argument("--dest_dir", type=str, required=True, help="Path of folder for Processor to save pooled result.")

    parser.add_argument("--train_pool_count", type=int, default=None, help="Num of pools to create for train data. Cannot be specified simultaneously with the `max_files_per_pool` parameter.")
    parser.add_argument("--test_pool_count", type=int, default=None, help="Num of pools to create for train data. Cannot be specified simultaneously with the `max_files_per_pool` parameter.")
    parser.add_argument("--max_files_per_pool", type=int, default=None, help="Max num of files in each pool. Cannot be specified simultaneously with the `train_pool_count` and `train_pool_count` parameters.")

    parser.add_argument("--train_test_ratio", type=float, default=0.8, help="Split ratio of train data and test data.")
    parser.add_argument("--split_mode", type=str, default="random", help="Mode of pool splitting, literlly `random` or `serial`. Default `random`.")
    parser.add_argument("--shutil_mode", type=str, default="", help="Mode of shutil operation, literlly `move` or `copy`. Default `move`.")

    return parser.parse_args()

# 20 file each pool
# train 110 test 28

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
    
    if not os.path.exists(args.dest_dir):    
        os.makedirs(args.dest_dir)

    logging.debug(f"Command: {' '.join(sys.argv)}")
    logging.debug(f"Params: {vars(args)}")

    processor = DataPoolProcessor(source_x_dir=args.source_x_dir,
                                  source_y_dir=args.source_y_dir,
                                  dest_dir=args.dest_dir,
                                  train_pool_count=args.train_pool_count,
                                  test_pool_count=args.test_pool_count,
                                  max_files_per_pool=args.max_files_per_pool,
                                  train_test_ratio=args.train_test_ratio,
                                  split_mode=args.split_mode,
                                  shutil_mode=args.shutil_mode)
    processor.process()

# python train_prepare.py --source_x_dir "C:\Users\C'heng\PycharmProjects\SWHY\data\preprocess\alpha_norm" --source_y_dir "C:\Users\C'heng\PycharmProjects\SWHY\data\preprocess\label_norm" --dest_dir "C:\Users\C'heng\PycharmProjects\SWHY\data\preprocess\pools_new" --max_files_per_pool 4 --train_test_ratio 0.8 --split_mode "random" --shutil_mode "copy" --log_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\preprocess\log"