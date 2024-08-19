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

from dataset import StockDataset, StockSequenceDataset, RandomSampleSampler
from nets import FactorVAE
from loss import ObjectiveLoss
from utils import str2bool

from torchviz import make_dot
from torchsummary import summary

def summary_wrapper(model:FactorVAE, num_stocks:int, num_features:int, seq_len:int):
    class WrapperModel(nn.Module):
        def __init__(self, model) -> None:
            super(WrapperModel, self).__init__()
            self.model:FactorVAE = model
        
        def forward(self, xy):
            x = xy
            y = x[-1,:,-1]

            return self.model(x, y)
    wrapper_model = WrapperModel(model)
    summary(wrapper_model, input_size=(num_stocks, num_features), batch_size=seq_len, device="cpu")

summary_wrapper(FactorVAE(101,2,32,32,46,0.1), 100, 101, 20)

