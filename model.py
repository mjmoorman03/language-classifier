from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset
import pandas as pd


class Model(torch.nn.Module):
    def __init__(self):
        self.numClasses = 2
        self.hiddenSize = 12
        self.inputSize = 1
        super(Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1)
        self.fc1 = torch.nn.Linear(self.hiddenSize, self.numClasses)
    

    def forward(self, x):
        x, _ = self.rnn(x)
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
            # make label 0 for Korean, 1 for English
            arrays.append(data[i]['audio']['array'])
            if data[i]['language'] == 'Korean':
                labels.append(0)
            elif data[i]['language'] == 'English':
                labels.append(1)
            else:
                raise ValueError('Invalid language')
        dataDict = {'data': arrays, 'labels': labels}
        self.data = pd.DataFrame(dataDict)

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


    
    
