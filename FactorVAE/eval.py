import os
import sys
import datetime
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

from tqdm import tqdm
from safetensors.torch import save_file, load_file
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard.writer import SummaryWriter

from matplotlib import pyplot as plt
import plotly.graph_objs as go

from dataset import StockDataset, StockSequenceDataset
from nets import FactorVAE
from loss import ObjectiveLoss, MSE_Loss, KL_Div_Loss, PearsonCorr, SpearmanCorr
from utils import str2bool


class FactorVAEEvaluator:
    def __init__(self,
                 model:FactorVAE,
                 device:torch.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        
        self.model:FactorVAE = model # FactorVAE 模型实例
        self.test_loader:DataLoader
        
        self.pred_eval_func:Union[nn.Module, Callable]
        self.latent_eval_func:Union[nn.Module, Callable] = KL_Div_Loss()
        self.pred_scores:List[float] = []
        self.latent_scores:List[float] = []
        
        self.y_true_list:List[torch.Tensor] = []
        self.y_hat_list:List[torch.Tensor] = []
        self.y_pred_list:List[torch.Tensor] = []
        
        self.log_folder:str = "log"
        self.device = device # 运算设备，默认为 CUDA（如果可用，否则为CPU）

        self.save_folder:str = "."
        self.plotter = Plotter()
    
    def load_dataset(self, test_set:StockSequenceDataset):
        self.test_loader = DataLoader(dataset=test_set,
                                        batch_size=None, 
                                        shuffle=False,
                                        num_workers=4)

    def load_checkpoint(self, model_path:str):
        if model_path.endswith(".pt"):
            self.model.load_state_dict(torch.load(model_path))
        elif model_path.endswith(".safetensors"):
            self.model.load_state_dict(load_file(model_path))
    
    def eval(self, metric:Literal["MSE", "IC", "Rank_IC"]="IC"):
        if metric == "MSE":
            self.pred_eval_func = MSE_Loss(scale=1)
        elif metric == "IC":
            self.pred_eval_func = PearsonCorr()
        elif metric == "Rank_IC":
            self.pred_eval_func = SpearmanCorr()
        
        self.eval_scores = []
        model = self.model.to(device=self.device)
        model.eval() # set eval mode to frozen layers like dropout
        with torch.no_grad(): 
            for batch, (X, y) in enumerate(tqdm(self.test_loader)):
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                y_pred, mu_posterior, sigma_posterior, mu_prior, sigma_prior = model(X, y)
                y_hat, mu_prior, sigma_prior  = model.predict(x=X)
                
                pred_score = self.pred_eval_func(y_hat, y)
                latent_score = self.latent_eval_func(mu_prior, sigma_prior, mu_posterior, sigma_posterior)
                
                self.pred_scores.append(pred_score.item())
                self.latent_scores.append(latent_score.item())
                
                self.y_true_list.append(y)
                self.y_hat_list.append(y_hat)
                self.y_pred_list.append(y_pred)
        logging.info(f"y pred score: {sum(self.pred_scores) / len(self.pred_scores)}")
        logging.info(f"latent kl divergence: {sum(self.latent_scores) / len(self.latent_scores)}")
    
    def visualize(self, idx:int=0):
        self.plotter.plot_score(self.pred_scores, 
                                self.latent_scores)
        self.plotter.save_fig(os.path.join(self.save_folder, "Scores"))
        self.plotter.plot_pred_sample(self.y_true_list, 
                                      self.y_hat_list, 
                                      self.y_pred_list,
                                      idx=idx)
        self.plotter.save_fig(os.path.join(self.save_folder, f"Trace {idx}"))

class Plotter:
    def __init__(self) -> None:
        pass
    
    def plot_score(self, pred_scores, latent_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(pred_scores, label='pred scores', marker='', color="b")
        plt.plot(latent_scores, label='latent scores', marker='', color="r")

        plt.legend()
        plt.title('Evaluation Scores')
        plt.xlabel('Index')
        plt.ylabel('Value')
    
    def plot_pred_sample(self, y_true_list, y_hat_list, y_pred_list, idx=0):
        y_true_list = [y_true[idx].item() for y_true in y_true_list]
        y_hat_list = [y_hat[idx].item() for y_hat in y_hat_list]
        y_pred_list = [y_pred[idx].item() for y_pred in y_pred_list]

        plt.figure(figsize=(10, 6))
        plt.plot(y_true_list, label='y true', marker='', color="g")
        plt.plot(y_hat_list, label='y rec', marker='', color="b")
        plt.plot(y_pred_list, label='y pred', marker='', color="r")

        plt.legend()
        plt.title('Comparison of y_true, y_hat, and y_pred')
        plt.xlabel('Index')
        plt.ylabel('Value')

    def save_fig(self, filename:str):
        if not filename.endswith(".png"):
            filename = filename + ".png"
        plt.savefig(filename)

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    datasets = torch.load(r"D:\PycharmProjects\SWHY\data\preprocess\dataset.pt")
    test_set = datasets["test"]

    model = FactorVAE(input_size=101, 
                      num_gru_layers=2, 
                      gru_hidden_size=32, 
                      hidden_size=16, 
                      latent_size=4,
                      gru_drop_out=0.1)
    
    evaluator = FactorVAEEvaluator(model=model)
    evaluator.load_checkpoint(r"D:\PycharmProjects\SWHY\model\factor-vae\model5\model5_epoch2.pt")
    evaluator.load_dataset(test_set)
    #print(trainer.model.feature_extractor.state_dict)
    #print(trainer.eval(test_set, "MSE"))
    
    evaluator.eval("MSE")
    evaluator.visualize()



                    
            
                

    

