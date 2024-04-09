from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        self.hiddenSize = 2
        self.inputSize = 1
        super(Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1, batch_first=False)
        self.fc1 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
    

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[-1]
        x = self.fc1(x)
        # not sure if softmax necessary or beneficial 
        x = torch.nn.functional.softmax(x, dim=1)
        return x
    

    def train(self, trainLoader):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(100):
            for i, data in enumerate(trainLoader, 0):
                trainData, trainLabels = data
                optimizer.zero_grad()
                outputs = self(trainData)
                # unclear on shape of outputs and trainLabels, if we need to take argmax of output first
                loss = criterion(outputs, trainLabels)
                loss.backward()
                optimizer.step()
            print('Epoch [%d/100], Loss: %.4f' % (epoch+1, loss.item()))


class AudioDataset(Dataset):
    def __init__(self, data):
        arrays = []
        labels = []
        for i in range(len(data)):
            if data[i]['audio']['sampling_rate'] != 16000:
                raise ValueError('Invalid sampling rate')
            arrays.append(torch.tensor(data[i]['audio']['array']).reshape(-1, 1, 1).type(torch.float32))
            # make label 0 for Korean, 1 for English
            if data[i]['language'] == 'Korean':
                labels.append(torch.tensor([0]))
            elif data[i]['language'] == 'English':
                labels.append(torch.tensor([1]))
            else:
                raise ValueError('Invalid language')
        self.data = list(zip(arrays, labels))
        # make about 80% of data training data
        trainData = self.data[:int(0.8*len(self.data))]
        testData = self.data[int(0.8*len(self.data)):]
        # batch data - need to make this work, may need to do it manually, and ought to 
        # pad data to make each sequence of the same length, at least within a batch
        self.trainLoader = torch.utils.data.DataLoader(trainData, batch_size=16, shuffle=True)
        self.testLoader = torch.utils.data.DataLoader(testData, batch_size=16, shuffle=True)
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def main():
    fleurs_korean = load_dataset('google/fleurs', "ko_kr", split='train', trust_remote_code=True)
    fleurs_english = load_dataset('google/fleurs', "en_us", split='train', trust_remote_code=True)
    data = concatenate_datasets([fleurs_korean, fleurs_english])
    data = AudioDataset(data)
    
    model = Model()
    model.train(data)
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()


    
    
