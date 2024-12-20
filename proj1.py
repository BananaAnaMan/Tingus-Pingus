import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

#torch.utils.data.DataLoader

#torch.utils.data.Dataset

#Device for Training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=lambda y: torch.zeroes(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
)




x = torch.rand(5, 3)
print(x)
print("hello World")

#Get a data set
#Analize the Data set
#program model to predict stuff about new input
#Develop front end website
