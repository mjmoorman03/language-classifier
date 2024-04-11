from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split



device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
    

BATCH_SIZE = 32

class Model(torch.nn.Module):
    def __init__(self):
        self.hiddenSize = 2
        self.inputSize = 1
        super(Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1, batch_first=True)
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
            for i, (trainData, trainLabels) in enumerate(trainLoader, 0):
                trainData, trainLabels = trainData.to(device), trainLabels.to(device)
                trainData = trainData.unsqueeze(2)
                optimizer.zero_grad()
                outputs = self(trainData)
                print(trainLabels.shape)
                print(outputs.shape)
                # unclear on shape of outputs and trainLabels, if we need to take argmax of output first
                loss = criterion(outputs, trainLabels)
                loss.backward()
                optimizer.step()
            print('Epoch [%d/100], Loss: %.4f' % (epoch+1, loss.item()))
    

def createDataloader(data):
    arrays = []
    labels = []
    maxLength = 0
    totalLength = 0
    for i in range(len(data)):
        maxLength = max(maxLength, len(data[i]['audio']['array']))
    for i in range(len(data)):
        if data[i]['audio']['sampling_rate'] != 16000:
            raise ValueError('Invalid sampling rate')
        tensor = torch.tensor(data[i]['audio']['array']).type(torch.float32)
        # pad data
        tensor = torch.nn.functional.pad(tensor, (0, maxLength - len(tensor)))
        arrays.append(tensor)
        # make label 0 for Korean, 1 for English
        if data[i]['language'] == 'Korean':
            labels.append(torch.tensor([0]))
        elif data[i]['language'] == 'English':
            labels.append(torch.tensor([1]))
        else:
            raise ValueError('Invalid language')
    # split data
    xTrain, xTest, yTrain, yTest = train_test_split(arrays, labels, test_size=0.15, random_state=42)
    xTrain = torch.stack(xTrain)
    xTest = torch.stack(xTest)
    yTrain = torch.stack(yTrain)
    yTest = torch.stack(yTest)
    trainData = torch.utils.data.TensorDataset(xTrain, yTrain) # type: ignore
    testData = torch.utils.data.TensorDataset(xTest, yTest) # type: ignore
        
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=32, shuffle=True) # type: ignore
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=32, shuffle=False) # type: ignore
    return (trainLoader, testLoader)


def main():
    fleurs_korean = load_dataset('google/fleurs', "ko_kr", split='train', trust_remote_code=True)
    fleurs_english = load_dataset('google/fleurs', "en_us", split='train', trust_remote_code=True)
    data = concatenate_datasets([fleurs_korean, fleurs_english]) # type: ignore
    trainLoader, testLoader = createDataloader(data)
    
    model = Model()
    model.to(device)
    model.train(trainLoader)
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()


    
    
