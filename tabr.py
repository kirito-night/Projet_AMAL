
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

class Model(nn.Module): 
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        #
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_main: int, # model dimension
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        def make_block(norm):
            if norm:
                return nn.Sequential(
                    nn.LayerNorm(d_main), 
                    nn.Linear(),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear()
                    )
            else: 
                return nn.Sequential(
                    nn.Linear(),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(), # ?
                    nn.Dropout(), # ?
                    )
        NE, NP = 1, 1
        self.block_E = [make_block(i>0) for i in range(NE)]
        self.block_P = [make_block(i>0) for i in range(NP)]
        self.P = nn.Sequential(
            nn.LayerNorm(),
            nn.ReLU(),
            nn.Linear()
        )
        #### R block ####
        self.S = nn.Linear() # x_dim
        self.V_y = nn.Linear()
        self.V_t = nn.Sequential(
            nn.Linear(bias = False),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear()
        )

    






