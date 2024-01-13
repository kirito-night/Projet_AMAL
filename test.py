import torch.nn.functional as F
import torch.utils.tensorboard
# from loguru import logger
from torch import Tensor
from tqdm import tqdm
import lib
import lib.data as data

dataset = data.build_dataset(
    seed = 0,
    cache = True,
    path = ":data/california",
    num_policy = "quantile",
    cat_policy = None,
    y_policy = "standard"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
dataset= dataset.to_torch(device)


if dataset.is_regression:
    dataset.data['Y'] = {k: v.float() for k, v in dataset.Y.items()}
Y_train = dataset.Y['train'].to(
    torch.long if dataset.is_multiclass else torch.float
)
Y_test = dataset.Y['test'].to(
    torch.long if dataset.is_multiclass else torch.float
)

