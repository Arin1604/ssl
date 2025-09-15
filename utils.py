import torch
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

plt.style.use('ggplot')

def get_data(batch_size=64):
    #wow so we can simply get the datasets using pytorch!?!?
    dataset_train = datasets.CIFAR10(root='data', train=True, download=True, transform=ToTensor() )
    dataset_validation = datasets.CIFAR10(root='data', train=False, download=True, transform=ToTensor() )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader

def plot_data(train_acc, valid_acc, train_loss, valid_loss, name=None):
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_accuracy.png'))

    plt.figure(figsize=(10,7))
    plt.plot(train_loss, color='tab:green', linestyle='-', label='train loss')
    plt.plot(valid_loss, color='tab:pink', linestyle='-', label='validation loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_loss.png'))




