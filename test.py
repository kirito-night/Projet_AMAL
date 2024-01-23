import torch.nn.functional as F
import torch.utils.tensorboard
# from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm
import lib
import lib.data as data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from oldtabr import Model
import delu

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


    

dataset = data.build_dataset(
    seed = 0,
    cache = True,
    path = ":data/california",
    num_policy = "quantile",
    cat_policy = None,
    y_policy = "standard"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset= dataset.to_torch(device)


if dataset.is_regression:
    dataset.data['Y'] = {k: v.float() for k, v in dataset.Y.items()}
Y_train = dataset.Y['train'].to(
    torch.long if dataset.is_multiclass else torch.float
)
Y_test = dataset.Y['test'].to(
    torch.long if dataset.is_multiclass else torch.float
)

BATCHSIZE = 10
n_epochs = 20
context_size = 10

train_size = dataset.size('train')
train_indices = torch.arange(train_size)

X_train = dataset.data['X_num']['train'].to(device)
X_test = dataset.data['X_num']['test'].to(device)


datatrain = MyDataset(X_train, Y_train)
datatest = MyDataset(X_test, Y_test)
loss_fn = nn.MSELoss()
progress = delu.ProgressTracker(16)

model = Model(
    n_num_features=dataset.n_num_features,
    n_bin_features=dataset.n_bin_features,
    cat_cardinalities=dataset.cat_cardinalities(),
    n_classes=dataset.n_classes(),
    #
    num_embeddings = None,  # lib.deep.ModuleSpec
    d_main =  265,
    d_multiplier = 2,
    encoder_n_blocks = 0,
    predictor_n_blocks = 1,
    mixer_normalization =  'auto',
    context_dropout= 0.389,
    dropout0= 0.389,
    dropout1=0,
    normalization= 'LayerNorm',
    activation= 'ReLU',
)

def make_mini_batch(data_size, batch_size, shuffle=True): 
    if shuffle:
        return torch.randperm(data_size).split(batch_size)
    else:
        return torch.aranges(data_size).split(batch_size)

def get_Xy(part, idx):
    batch = (
        {
            key[2:]: dataset.data[key][part]
            for key in dataset.data
            if key.startswith('X_')
        },
        dataset.Y[part],
    )
    return (
        batch
        if idx is None
        else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
    )

def apply_model(part: str, idx: Tensor, training: bool):
    x, y = get_Xy(part, idx)

    candidate_indices = train_indices
    is_train = part == 'train'
    if is_train:
        # NOTE: here, the training batch is removed from the candidates.
        # It will be added back inside the model's forward pass.
        candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
    candidate_x, candidate_y = get_Xy(
        'train',
        # This condition is here for historical reasons, it could be just
        # the unconditional `candidate_indices`.
        None if candidate_indices is train_indices else candidate_indices,
    )

    return model(
        x_=x,
        y=y if is_train else None,
        candidate_x_=candidate_x,
        candidate_y=candidate_y,
        context_size=context_size,
        is_train=is_train,
    ).squeeze(-1)

epoch = 0
while not progress.fail:
    for idx in make_mini_batch(train_size, batch_size=BATCHSIZE):
        model.train()
        print(idx.shape)
        prediction = apply_model('train', idx, training=True)
        print('ok')
        break
    epoch += 1
    break



