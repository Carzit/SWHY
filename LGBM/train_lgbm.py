"""
LGBM模型训练模块
由于数据量较大，我们将数据分池后分别训练LGBM模型，并通过模型集成器将结果进行集成。
以实现分布式训练。
"""

import os
import sys
import json
import pickle
import logging
import argparse
from numbers import Number
from typing import Union, Dict, List, Any, Literal, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

def load_pool_data(data_dir:str) -> pd.DataFrame:
    # 从指定目录中读取所有CSV文件，并将它们合并成一个DataFrame。
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_list = [pd.read_csv(f, index_col=[0,1,2]) for f in files]
    return pd.concat(df_list, ignore_index=True)
        
class LGBMTrainer:
    """
    LightGBM模型训练管线
    """
    def __init__(self,
                 train_x_dir:str,
                 train_y_dir:str,
                 label_name:str,
                 test_x_dir:Optional[str]=None,
                 test_y_dir:Optional[str]=None,
                 model_path:Optional[str]=None, 
                 metric:Literal["MSE", "RMSE", "IC", "Rank_IC"]="IC"):

        self.train_x_dir:str = train_x_dir
        self.train_y_dir:str = train_y_dir
        self.test_x_dir:str = test_x_dir
        self.test_y_dir:str = test_y_dir
        self.label_name = label_name

        self.model:lgb.LGBMModel = None
        self.model_path:str = model_path

        self.metric:str = metric
        self.eval_score:float = None

    def load_pool_data(self, data_dir:str) -> pd.DataFrame:
        # 与全局的load_pool_data函数相同，用于加载指定目录中的CSV文件。
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        df_list = [pd.read_csv(f, index_col=[0,1,2]) for f in files]
        return pd.concat(df_list, ignore_index=True)
    
    def set_params(self,
                   model_type: Literal["regressor", "cluster", "ranker"] = "regressor",
                   boosting_type: Literal["gbdt", "dart", "goss", "rf"] = "gbdt", 
                   num_leaves: int = 31, 
                   max_depth: int = -1, 
                   learning_rate: float = 0.1, 
                   n_estimators: int = 100, 
                   subsample_for_bin: int = 200000, 
                   objective: str | None = None, 
                   class_weight: Dict | str | None = None, 
                   min_split_gain: float = 0, 
                   min_child_weight: float = 0.001, 
                   min_child_samples: int = 20, 
                   subsample: float = 1, 
                   subsample_freq: int = 0, 
                   colsample_bytree: float = 1, 
                   reg_alpha: float = 0, 
                   reg_lambda: float = 0, 
                   random_state = None, 
                   n_jobs: int | None = None, 
                   importance_type: str = "split"):
        # 设置LightGBM模型的参数。

        if model_type == "regressor":
            model_class = lgb.LGBMRegressor
        elif model_type == "cluster":
            model_class = lgb.LGBMClassifier
        elif model_type == "ranker":
            model_class = lgb.LGBMRanker
        else:
            raise NotImplementedError()   
    
        self.model = model_class(boosting_type=boosting_type,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 subsample_for_bin=subsample_for_bin,
                                 objective=objective,
                                 class_weight=class_weight,
                                 min_split_gain=min_split_gain,
                                 min_child_weight=min_child_weight,
                                 min_child_samples=min_child_samples,
                                 subsample=subsample,
                                 subsample_freq=subsample_freq,
                                 colsample_bytree=colsample_bytree,
                                 reg_alpha=reg_alpha,
                                 reg_lambda=reg_lambda,
                                 random_state=random_state,
                                 n_jobs=n_jobs,
                                 importance_type=importance_type)
    
    
    def save_model(self):
        # 将训练好的模型保存到指定路径。
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)  

    def eval(self, 
             metric:Literal["MSE", "RMSE", "IC", "Rank_IC", None]=None,         
             test_x_dir:Optional[str] = None,
             test_y_dir:Optional[str] = None) -> float:
        # 评估模型在测试集上的性能。

        if test_x_dir is not None:
            self.test_x_dir = test_x_dir
        if test_y_dir is not None:
            self.test_y_dir = test_y_dir

        test_x = self.load_pool_data(self.test_x_dir)
        test_y = self.load_pool_data(self.test_y_dir)[self.label_name]

        y_pred = self.model.predict(test_x.values, num_iteration=self.model._best_iteration)

        if metric is None:
            metric = self.metric
        else:
            self.metric = metric
        if metric == "MSE":
            eval_score = ((y_pred - test_y.values) ** 2).mean()
        elif  metric == "RMSE":
            eval_score = ((y_pred - test_y.values) ** 2).mean() ** 0.5
        elif  metric == "IC":
            eval_score, _ = pearsonr(y_pred, test_y.values)
        elif  metric == "Rank_IC":
            eval_score, _ = spearmanr(y_pred, test_y.values)
        else:
            raise NotImplementedError()
        return eval_score
    
    def train(self):
        # 训练LightGBM模型，并将训练好的模型保存。
        train_x = self.load_pool_data(self.train_x_dir)
        train_y = self.load_pool_data(self.train_y_dir)[self.label_name]
        self.model.fit(X=train_x.values, y=train_y.values)
        self.save_model()

class DistributedTrainer:
    """
    LGBM分布式训练管线。支持回归、聚类和排序任务。
    """
    def __init__(self,
                 data_folder:str,
                 label_name:str,
                 model_folder:Optional[str]=None, 
                 metric:Literal["MSE", "RMSE", "IC", "Rank_IC"]="IC"):

        self.data_folder = data_folder
        with open(os.path.join(data_folder, "file_structure.json"), "r") as f:
            self.file_structure = json.load(f)
        self.label_name = label_name

        self.model:lgb.LGBMModel = None
        if model_folder is None:
            self.model_folder:str = os.path.join(data_folder, "models")
            os.makedirs(self.model_folder, exist_ok=True)
        else:
            self.model_folder:str = model_folder

        self.metric:str = metric
        self.eval_scores:Dict[str, List[float]] = dict()

        self.model_type: Literal["regressor", "cluster", "ranker"] = "regressor"
        self.boosting_type: Literal["gbdt", "dart", "goss", "rf"] = "gbdt"
        self.num_leaves: int = 31
        self.max_depth: int = -1
        self.learning_rate: float = 0.1
        self.n_estimators: int = 100
        self.subsample_for_bin: int = 200000
        self.objective: str | None = None
        self.class_weight: Dict | str | None = None
        self.min_split_gain: float = 0
        self.min_child_weight: float = 0.001
        self.min_child_samples: int = 20
        self.subsample: float = 1
        self.subsample_freq: int = 0
        self.colsample_bytree: float = 1
        self.reg_alpha: float = 0
        self.reg_lambda: float = 0
        self.random_state = None
        self.n_jobs: int | None = None
        self.importance_type: str = "split"

        self.ensemble_model: ModelEnsembler = None
    
    def set_params(self,
                   model_type: Literal["regressor", "cluster", "ranker"] = "regressor",
                   boosting_type: Literal["gbdt", "dart", "goss", "rf"] = "gbdt", 
                   num_leaves: int = 31, 
                   max_depth: int = -1, 
                   learning_rate: float = 0.1, 
                   n_estimators: int = 100, 
                   subsample_for_bin: int = 200000, 
                   objective: str | None = None, 
                   class_weight: Dict | str | None = None, 
                   min_split_gain: float = 0, 
                   min_child_weight: float = 0.001, 
                   min_child_samples: int = 20, 
                   subsample: float = 1, 
                   subsample_freq: int = 0, 
                   colsample_bytree: float = 1, 
                   reg_alpha: float = 0, 
                   reg_lambda: float = 0, 
                   random_state = None, 
                   n_jobs: int | None = None, 
                   importance_type: str = "split"):
        # 设置模型参数
        self.model_type = model_type
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate 
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin 
        self.objective = objective
        self.class_weight = class_weight 
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight 
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq 
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda 
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type

    def train(self):
        # 训练 LightGBM 模型，并保存到 model_folder；对每个训练池（train pool）进行训练，并评估其在测试池（test pool）上的表现。
        logging.debug(f"Train pools: {len(self.file_structure['train_data']['X'].keys())}; Test pools: {len(self.file_structure['test_data']['X'].keys())}")
        logging.debug("Start training")
        for train_pool in tqdm(self.file_structure["train_data"]["X"].keys()):
            train_x_dir = os.path.join(self.data_folder, "train_data", "X", train_pool)
            train_y_dir = os.path.join(self.data_folder, "train_data", "Y", train_pool)

            trainer = LGBMTrainer(train_x_dir, 
                                  train_y_dir,  
                                  label_name=self.label_name, 
                                  model_path=os.path.join(self.model_folder, f"{train_pool}.pkl"),
                                  metric=self.metric)
            trainer.set_params(
                model_type=self.model_type,
                boosting_type = self.boosting_type,
                num_leaves = self.num_leaves,
                max_depth = self.max_depth,
                learning_rate = self.learning_rate,
                n_estimators = self.n_estimators,
                subsample_for_bin = self.subsample_for_bin, 
                objective = self.objective,
                class_weight = self.class_weight,
                min_split_gain = self.min_split_gain,
                min_child_weight = self.min_child_weight, 
                min_child_samples = self.min_child_samples,
                subsample = self.subsample,
                subsample_freq = self.subsample_freq, 
                colsample_bytree = self.colsample_bytree,
                reg_alpha = self.reg_alpha,
                reg_lambda = self.reg_lambda, 
                random_state = self.random_state,
                n_jobs = self.n_jobs,
                importance_type = self.importance_type)
            
            trainer.train()

            self.eval_scores[train_pool] = []
            for test_pool in self.file_structure["test_data"]["X"].keys():
                test_x_dir = os.path.join(self.data_folder, "test_data", "X", test_pool)
                test_y_dir = os.path.join(self.data_folder, "test_data", "Y", test_pool)
                eval_score = trainer.eval(test_x_dir=test_x_dir, test_y_dir=test_y_dir)
                self.eval_scores[train_pool].append(eval_score)
            logging.info(f"AVG {self.metric} for train_{train_pool} model: {sum(self.eval_scores[train_pool]) / len(self.eval_scores[train_pool])}")
        logging.debug(f"Training complete and models saved to {self.model_folder}")

    def eval(self, ensemble_model):
        # 使用集成模型对测试数据进行评估，并计算平均评估分数。
        self.ensemble_model = ensemble_model
        ensemble_eval_scores = []
        for test_pool in self.file_structure["test_data"]["X"].keys():
            test_x_dir = os.path.join(self.data_folder, "test_data", "X", test_pool)
            test_y_dir = os.path.join(self.data_folder, "test_data", "Y", test_pool)
            test_x = load_pool_data(test_x_dir).values
            test_y = load_pool_data(test_y_dir)[self.label_name].values
            ensemble_eval_score = self.ensemble_model.eval(test_x, test_y, metric=self.metric)
            ensemble_eval_scores.append(ensemble_eval_score)
        logging.info(f"Ensemble Total AVG {self.metric}: {sum(ensemble_eval_scores) / len(ensemble_eval_scores)}")
        return sum(ensemble_eval_scores) / len(ensemble_eval_scores)


class ModelEnsembler:
    """
    模型集成器的基类，定义了模型集成的基本框架和功能。
    """
    def __init__(self, model_paths:Optional[List[str]]=None, model_folder:Optional[str]=None):
        self.model_paths: List[str]
        self.models: List[lgb.LGBMModel] = []

        # 如果提供了model_paths，则直接使用；如果提供了model_folder，则从该文件夹中读取所有以.pkl结尾的文件作为模型文件路径。如果两者都没有提供，则抛出ValueError。
        if model_paths is not None:
            self.model_paths = model_paths
        elif model_folder is not None:
            self.model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.pkl')]
        else:
            raise ValueError("model_paths and model_folder can be specified either but not both.")

    def load_model(self):
        # 加载模型。遍历self.model_paths，从每个路径加载模型，并将其添加到self.models列表中。
        for model_path in self.model_paths:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            self.models.append(model)

    def predict(self, X):
        # 预测方法。这是一个抽象方法，需要在子类中实现。用于对输入数据进行预测
        raise NotImplementedError()

    def eval(self, X, y, metric:Literal["MSE", "RMSE", "IC", "Rank_IC"]="IC"):
        # 评估方法。对预测结果进行评估，支持多种评估指标（MSE、RMSE、IC、Rank_IC）。
        y_pred = self.predict(X)

        if metric == "MSE":
            eval_score = ((y_pred - y) ** 2).mean()
        elif metric == "RMSE":
            eval_score = ((y_pred - y) ** 2).mean() ** 0.5
        elif metric == "IC":
            eval_score, _ = pearsonr(y_pred, y)
        elif metric == "Rank_IC":
            eval_score, _ = spearmanr(y_pred, y)
        else:
            raise NotImplementedError()
        return eval_score

class AverageModelEnsembler(ModelEnsembler):
    """继承自ModelEnsembler，实现了一种简单的模型集成方法：平均法。"""
    def __init__(self, model_paths: List[str] | None = None, model_folder: str | None = None):
        super().__init__(model_paths, model_folder)

    def predict(self, X) -> np.ndarray:
        preds = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        return preds.mean(axis=1)
    
class WeightedModelEnsembler(ModelEnsembler):
    """继承自ModelEnsembler，实现了一种加权平均的模型集成方法。需要在初始化时传入权重。"""
    def __init__(self,
                 weights: List[Number],
                 model_paths: List[str] | None = None, 
                 model_folder: str | None = None):
        super().__init__(model_paths, model_folder)
        self.weights:List[float] = [weight / sum(weights) for weight in weights]
    
    def load_model(self):
        super().load_model()
        assert len(self.weights) != len(self.models), f"weights and models not match: weight num is {len(self.weights)} but model num is {len(self.models)}"

    def predict(self, X) -> np.ndarray:
        preds = np.zeros((X.shape[0], len(self.models)))
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            preds[:, i] = model.predict(X) * weight
        return preds.mean(axis=1)

class StackingModelEnsembler(ModelEnsembler):
    """继承自ModelEnsembler，实现了一种堆叠（Stacking）的模型集成方法。具体地，在训练集上额外训练一个无截距的线性模型来决定各个模型的权重。"""
    def __init__(self,
                 model_paths: List[str] | None = None, 
                 model_folder: str | None = None):
        super().__init__(model_paths, model_folder)
        self.meta_model = LinearRegression(fit_intercept=False)
    
    @property
    def coef(self):
        return self.meta_model.coef_
    
    @property
    def intercept(self):
        return self.meta_model.intercept_

    def fit(self, X, y):
        base_preds = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            base_preds[:, i] = model.predict(X)
        self.meta_model.fit(base_preds, y)

    def predict(self, X) -> np.ndarray:
        base_preds = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            base_preds[:, i] = model.predict(X)
        return self.meta_model.predict(base_preds)

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Data Normalizer.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--data_folder", type=str, required=True, help="Path of folder for train data files")
    parser.add_argument("--label_name", type=str, required=True, help="Target label name (col name in Y files)")
    parser.add_argument("--model_folder", type=str, default=None, help="Path of folder for Trainer to save models. If not specified, `model` folder will be created under data_folder.")
    parser.add_argument("--metric", type=str, default="IC", help="Eval metric type, literally `MSE`, `RMSE`, `IC` or `Rank_IC`. Default `IC`. ")

    parser.add_argument("--model_type", type=str, default="regressor", help="Model type, literally `regressor`, `cluster`or `ranker`. Default `regressor`.")
    parser.add_argument("--boosting_type", type=str, default="gbdt", help="Boosting type, literally `gbdt`, `dart`, `goss` or ``rf`. Default `gbdt`.")
    parser.add_argument("--num_leaves", type=int, default=31, help="Maximum tree leaves for base learners. Default 31")
    parser.add_argument("--max_depth", type=int, default=-1, help="Maximum tree depth for base learners, -1 means no limit. Default -1")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Boosting learning rate. Default 0.1")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of boosted trees to fit. Default 100")
    parser.add_argument("--subsample_for_bin", type=int, default=200000, help="Number of bucketed bins for feature values. Default 200,000")
    parser.add_argument("--min_split_gain", type=float, default=0, help="Minimum loss reduction required to make a further partition on a leaf node of the tree. Default 0")
    parser.add_argument("--min_child_weight", type=float, default=0.001, help="Minimum sum of instance weight(hessian) needed in a child(leaf). Default 0.001")
    parser.add_argument("--min_child_samples", type=int, default=20, help="Minimum number of data need in a child(leaf). Default 20")
    parser.add_argument("--subsample", type=float, default=1, help="Subsample ratio of the training instance. Default 1")
    parser.add_argument("--subsample_freq", type=int, default=0, help="Frequence of subsample, <=0 means no enable. Default 0")
    parser.add_argument("--colsample_bytree", type=float, default=1, help="Subsample ratio of columns when constructing each tree. Default 1")
    parser.add_argument("--reg_alpha", type=float, default=0, help="L1 regularization term on weights. Default 0")
    parser.add_argument("--reg_lambda", type=float, default=0, help="L2 regularization term on weights. Default 0")
    parser.add_argument("--random_state", type=int, default=None, help="Random number seed. Will use default seeds in c++ code if set to None. Default None.")

    parser.add_argument("--ensemble_type", type=str, default="average", help="Model ensemble type, literally `average`, `weighted` or `stacking`. Default `average`.")
    parser.add_argument("--weights", type=float, default=None, nargs="+", help="Model ensemble weights, necessary if ensemble_type is `weighted`.")

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
    
    trainer = DistributedTrainer(data_folder=args.data_folder,
                                 label_name=args.label_name,
                                 model_folder=args.model_folder,
                                 metric=args.metric)
    trainer.set_params(
                model_type=args.model_type,
                boosting_type = args.boosting_type,
                num_leaves = args.num_leaves,
                max_depth = args.max_depth,
                learning_rate = args.learning_rate,
                n_estimators = args.n_estimators,
                subsample_for_bin = args.subsample_for_bin, 
                min_split_gain = args.min_split_gain,
                min_child_weight = args.min_child_weight, 
                min_child_samples = args.min_child_samples,
                subsample = args.subsample,
                subsample_freq = args.subsample_freq, 
                colsample_bytree = args.colsample_bytree,
                reg_alpha = args.reg_alpha,
                reg_lambda = args.reg_lambda, 
                random_state = args.random_state)
    trainer.train()

    if args.ensemble_type == "average":
        ensembler = AverageModelEnsembler(model_folder=trainer.model_folder)
        ensembler.load_model()
    elif args.ensemble_type == "weighted":
        ensembler = WeightedModelEnsembler(weights=args.weights, model_folder=trainer.model_folder)
        ensembler.load_model()
    elif args.ensemble_type == "stacking":
        ensembler = StackingModelEnsembler(model_folder=trainer.model_folder)
        ensembler.load_model()
        for train_pool in trainer.file_structure["train_data"]["X"].keys():
            train_x_dir = os.path.join(trainer.data_folder, "train_data", "X", train_pool)
            train_y_dir = os.path.join(trainer.data_folder, "train_data", "Y", train_pool)

            train_x = trainer.load_pool_data(trainer.train_x_dir)
            train_y = trainer.load_pool_data(trainer.train_y_dir)[trainer.label_name]

            ensembler.fit(X=train_x.values, y=train_y.values)

    trainer.eval(ensembler)

# python train_lgbm.py --data_folder "data\preprocess\pool" --label_name "ret10" --model_folder "model\lgbm" --metric "IC" --ensemble_type "average" --random_state 123 --log_folder "log"