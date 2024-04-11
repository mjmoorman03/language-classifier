from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
import scipy.signal as signal

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
    

BATCH_SIZE = 32

class Model(torch.nn.Module):
    def __init__(self):
        self.hiddenSize = 2
        self.inputSize = 1
        super(Model, self).__init__()
        self.cnn = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(16000, 2)
    

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 16000)
        x = self.fc1(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x
    

    def train(self, trainLoader):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(100):
            for i, (trainData, trainLabels) in enumerate(trainLoader, 0):
                trainData, trainLabels = trainData.to(device), trainLabels.to(device)
                optimizer.zero_grad()
                outputs = self(trainData)
                print(trainLabels.shape)
                print(outputs.shape)
                # unclear on shape of outputs and trainLabels, if we need to take argmax of output first
                loss = criterion(outputs, trainLabels)
                loss.backward()
                optimizer.step()
            print('Epoch [%d/100], Loss: %.4f' % (epoch+1, loss.item()))


def plotSpectrogram(audio, samplingRate):
    f, t, Sxx = signal.spectrogram(audio, samplingRate)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
