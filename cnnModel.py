from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, spectrogram, ShortTimeFFT
import json
import audioInputUserChoice as audioInput


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

BATCH_SIZE = 16
NUM_EPOCHS = 10
MAX_LENGTH = 16000 * 11
LEARNING_RATE = 0.001
NUM_FILTERS = 16
NUM_SECOND_FILTERS = 32
NUM_THIRD_FILTERS = 64
NUM_FOURTH_FILTERS = 128
LANGUAGES = ['Korean', 'English', 'Spanish']

class Model(torch.nn.Module):
    def __init__(self, numLanguages):
        super(Model, self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=NUM_FILTERS, kernel_size=3, stride=1, padding=1)
        self.cnn2 = torch.nn.Conv2d(in_channels=NUM_FILTERS, out_channels=NUM_SECOND_FILTERS, kernel_size=3, stride=1, padding=1)
        self.cnn3 = torch.nn.Conv2d(in_channels=NUM_SECOND_FILTERS, out_channels=NUM_THIRD_FILTERS, kernel_size=3, stride=1, padding=1)
        self.cnn4 = torch.nn.Conv2d(in_channels=NUM_THIRD_FILTERS, out_channels=NUM_FOURTH_FILTERS, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(1250 * NUM_FOURTH_FILTERS, numLanguages)
        self.batchNorm1 = torch.nn.BatchNorm2d(NUM_FILTERS)
        self.batchNorm2 = torch.nn.BatchNorm2d(NUM_SECOND_FILTERS)
        self.batchNorm3 = torch.nn.BatchNorm2d(NUM_THIRD_FILTERS)
        self.batchNorm4 = torch.nn.BatchNorm2d(NUM_FOURTH_FILTERS)
    

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.nn.functional.relu(x)
        x = self.batchNorm1(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.cnn2(x)
        x = torch.nn.functional.relu(x)
        x = self.batchNorm2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.cnn3(x)
        x = torch.nn.functional.relu(x)
        x = self.batchNorm3(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.cnn4(x)
        x = torch.nn.functional.relu(x)
        x = self.batchNorm4(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x
    

    def train(self, trainLoader):
        losses = []
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(NUM_EPOCHS):
            for _, (trainData, trainLabels) in enumerate(trainLoader, 0):
                trainData, trainLabels = trainData.to(device), trainLabels.to(device)
                optimizer.zero_grad()
                trainData = trainData.unsqueeze(1)
                outputs = self(trainData)
                trainLabels = trainLabels.squeeze(1)
                # unclear on shape of outputs and trainLabels, if we need to take argmax of output first
                loss = criterion(outputs, trainLabels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            print(f'Epoch [%d/{NUM_EPOCHS}], Loss: %.4f' % (epoch+1, loss.item()))
        return losses

        
    def test(self, testLoader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testLoader:
                testInputs, testLabels = data
                testInputs, testLabels = testInputs.to(device), testLabels.to(device)
                testInputs = testInputs.unsqueeze(1)
                outputs = self(testInputs)
                predicted = torch.argmax(outputs, 1)
                testLabels = testLabels.squeeze(1)
                total += testLabels.size(0)
                correct += (predicted == testLabels).sum().item()
        print(f'Accuracy of the network on the test data: {100 * correct/total}')
        return 100 * correct / total


def transformAudio(audio, samplingRate):
    nperseg = 4000
    noverlap = 2000
    win = ('gaussian', 1e-2 * samplingRate)
    SFT = ShortTimeFFT.from_window(win, samplingRate, nperseg, noverlap, fft_mode='centered',
                               scale_to='magnitude', phase_shift=None)
    transformed = SFT.stft(audio)
    return transformed


def plotSpectrogramFromFFT(audio, samplingRate):
    # apply activation function to audio    
    nperseg = 4000
    noverlap = 2000
    win = ('gaussian', 1e-2 * samplingRate)
    SFT = ShortTimeFFT.from_window(win, samplingRate, nperseg, noverlap, fft_mode='centered',
                               scale_to='magnitude', phase_shift=None)
    transformed = SFT.stft(audio)
    N = len(audio)
    fig1, axx = plt.subplots(1, 1, sharex='all', sharey='all',
                            figsize=(6., 5.))  # enlarge figure a bit
    axx.set_title(r"ShortTimeFFT produces $%d\times%d$ points" % transformed.T.shape)
    axx.set_xlabel(rf"Time $t$ in seconds ($\Delta t= {SFT.delta_t:g}\,$s)")
    # Calculate extent of plot with centered bins since
    # imshow does not interpolate by default:
    extent1 = SFT.extent(N, center_bins=True)

    kw = dict(origin='lower', aspect='auto', cmap='viridis')
    im1b = axx.imshow(abs(transformed), extent=extent1, **kw)
    fig1.colorbar(im1b, ax=axx, label="Magnitude $|S_z(t, f)|$")
    _ = fig1.supylabel(r"Frequency $f$ in Hertz ($\Delta f = %g\,$Hz)" %
                    SFT.delta_f, x=0.08, y=0.5, fontsize='medium')
    plt.show()


def createDataloader(data):
    arrays = []
    labels = []
    # max seconds 
    for i in range(len(data)):
        # pad or truncate data to 11 seconds
        tensor = torch.tensor(data[i]['audio']['array']).type(torch.float32)
        if len(tensor) > MAX_LENGTH:
            tensor = tensor[:MAX_LENGTH]
        elif MAX_LENGTH - len(tensor) > 0:
            tensor = torch.nn.functional.pad(tensor, (0, MAX_LENGTH - len(tensor)))
        transformed = transformAudio(tensor, data[i]['audio']['sampling_rate'])
        arrays.append(torch.tensor(transformed).type(torch.float32))
        # labels for languages to indices
        if data[i]['language'] in LANGUAGES:
            labels.append(torch.tensor([LANGUAGES.index(data[i]['language'])]))
        else:
            raise ValueError('Invalid language')      
       
    xTrain, xTest, yTrain, yTest = train_test_split(arrays, labels, test_size=0.15, random_state=None)
    xTrain = torch.stack(xTrain)
    xTest = torch.stack(xTest)
    yTrain = torch.stack(yTrain)
    yTest = torch.stack(yTest)
    trainData = torch.utils.data.TensorDataset(xTrain, yTrain) # type: ignore
    testData = torch.utils.data.TensorDataset(xTest, yTest) # type: ignore
        
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True) # type: ignore
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False) # type: ignore
    return (trainLoader, testLoader)


def graphLoss(losses):
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def saveModelAccuracy(model, accuracy):
    langs = ''
    for lang in LANGUAGES:
        langs += lang + '_'
    # open accuracies.json and save accuracy if higher than previous
    with open('accuracies.json', 'r') as f:
        data = json.load(f)
    if langs not in data.keys() or float(data[langs]) < accuracy:
        torch.save(model.state_dict(), f'{langs}model.pth')
        data[langs] = str(accuracy)
        with open('accuracies.json', 'w') as f:
            json.dump(data, f)


def loadModel(languages=LANGUAGES):
    model = Model(len(languages))
    pathName = ''
    for lang in languages:
        pathName += lang + '_'
    model.to(device)
    model.load_state_dict(torch.load(f'{pathName}model.pth'))
    return model


def evaluateAudio(model, audio, samplingRate):
    audio = torch.tensor(audio).type(torch.float32)
    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]
    elif MAX_LENGTH - len(audio) > 0:
        audio = torch.nn.functional.pad(audio, (0, MAX_LENGTH - len(audio)))
    transformed = transformAudio(audio, samplingRate)
    transformed = torch.tensor(transformed).type(torch.float32)
    transformed = transformed.unsqueeze(0).unsqueeze(1)
    print(transformed.shape)
    plotSpectrogramFromFFT(audio, samplingRate)
    transformed = transformed.to(device)
    outputs = model(transformed)
    print(outputs)
    predicted = torch.argmax(outputs, 1)
    return LANGUAGES[predicted]


def classifyMicrophoneInput(model):
    audio = audioInput.audioInputUserChoice()
    return evaluateAudio(model, audio, 16000)


def saveResults(model, accuracy):
    langs = ''
    for lang in LANGUAGES:
        langs += lang + '_'
    # open results.json and save model summary and accuracy
    with open('results.json', 'r+') as f:
        data = json.load(f)
        index = len(data)
        data[index] = {}
        data[index]['model'] = str(model)
        data[index]['accuracy'] = str(accuracy)
        data[index]['languages'] = langs
        f.seek(0)
        json.dump(data, f)


def mainTrain():
    fleurs_korean = load_dataset('google/fleurs', "ko_kr", split='train', trust_remote_code=True) # type: ignore
    fleurs_english = load_dataset('google/fleurs', "en_us", split='train', trust_remote_code=True) 
    fleurs_spanish = load_dataset('google/fleurs', "es_419", split='train', trust_remote_code=True)
    data = concatenate_datasets([fleurs_korean, fleurs_english, fleurs_spanish]) # type: ignore
    trainLoader, testLoader = createDataloader(data)
    
    model = Model(len(LANGUAGES))
    model.to(device)
    model.train(trainLoader)
    accuracy = model.test(testLoader)
    saveModelAccuracy(model, accuracy)
    saveResults(model, accuracy)


def main():
    mainTrain()


if __name__ == '__main__':
    main()

