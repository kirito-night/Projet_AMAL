
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
# from loguru import logger
from torch import Tensor
from tqdm import tqdm

import lib


dataset = lib.build_dataset(
    name="mnist",
    path="data",
    download=True,
    normalize=True,
    flatten=True,
    one_hot=False,
    train_size=0.8,
    random_state=42,
    verbose=True,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
if dataset.is_regression:
    dataset.data['Y'] = {k: v.float() for k, v in dataset.Y.items()}
Y_train = dataset.Y['train'].to(
    torch.long if dataset.is_multiclass else torch.float
)
Y_test = dataset.Y['test'].to(
    torch.long if dataset.is_multiclass else torch.float
)
X_train = dataset.X['train'].to(device)
X_test = dataset.X['test'].to(device)

for k, v in dataset.X.items():
    print(k, v.shape)
for k, v in dataset.Y.items():
    print(k, v.shape)
    


