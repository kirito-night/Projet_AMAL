import torch.nn.functional as F
import torch.utils.tensorboard
# from loguru import logger
from torch import Tensor
from tqdm import tqdm
import lib
import lib.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np

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

BATCHSIZE = 128

X_train = dataset.data['X_num']['train'].to(device)
X_test = dataset.data['X_num']['test'].to(device)


datatrain = MyDataset(X_train, Y_train)
datatest = MyDataset(X_test, Y_test)

trainloader = torch.utils.data.DataLoader(datatrain, batch_size=BATCHSIZE,shuffle=True)
testloader = torch.utils.data.DataLoader(datatest , batch_size=BATCHSIZE ,shuffle=True)


for x,y in trainloader:
    print(y)

# def train_epoch(dataLoader, model, loss_fn, optim, device = None, num_classes = 2):
#     if device == None: 
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     allLoss = []
#     for x,y in tqdm(dataLoader):
#         optim.zero_grad()
#         # for scores
#         acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
#         x = x.to(device)
#         y = y.to(device)
#         yhat = model(x)
#         loss = loss_fn(yhat,y)
#         allLoss.append(loss.item())
#         acc(yhat.argmax(1), y)

#         # Optimization
#         loss.backward()
#         optim.step()
        
    
#     return np.array(allLoss).mean(), acc.compute().item()

# # check loss on test and accuracy
# def test_epoch(dataLoader, model, loss_fn, device = None, num_classes = 2):
#     if device == None: 
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     allLoss = []
#     with torch.no_grad():
#         for x,y in tqdm(dataLoader):
#             # for scores
#             acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
#             x = x.to(device)
#             y = y.to(device)
#             yhat = model(x)
#             loss = loss_fn(yhat,y)
#             allLoss.append(loss.item())
#             acc(yhat.argmax(1), y)

#     return np.array(allLoss).mean(), acc.compute().item()
    


# eSize = 50
# outputS = 2


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Model(embeddings, outputS, eSize).to(device)

# # Loss function
# loss_fn = nn.

# # Optimisateur 
# lr = 0.001
# optim = torch.optim.Adam(list(model.parameters()),lr = lr)
# optim.zero_grad()

# num_epoch = 7
# for epoch in range(num_epoch):
#     print("Epoch :",epoch)
#     resultLoss, trainAcc = train_epoch(trainloader,model,loss_fn,optim,device,num_classes=outputS)
#     testLoss, testAcc = test_epoch(testloader, model, loss_fn, device=device, num_classes=outputS)
#     print(f'Training Loss : {resultLoss} and Accuracy {trainAcc:.2f}')
#     print(f'Test Loss : {testLoss} and Accuracy {testAcc:.2f}')
