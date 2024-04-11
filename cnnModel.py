from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, spectrogram, ShortTimeFFT
import random

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
    

BATCH_SIZE = 32

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.cnn2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(22000, 2)
    

    def forward(self, x):
        x = self.cnn1(x)
        # max pooling
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.cnn2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x
    

    def train(self, trainLoader):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(100):
            for i, (trainData, trainLabels) in enumerate(trainLoader, 0):
                trainData, trainLabels = trainData.to(device), trainLabels.to(device)
                optimizer.zero_grad()
                trainData = trainData.unsqueeze(1)
                outputs = self(trainData)
                trainLabels = trainLabels.squeeze(1)
                # unclear on shape of outputs and trainLabels, if we need to take argmax of output first
                loss = criterion(outputs, trainLabels)
                loss.backward()
                optimizer.step()
            print('Epoch [%d/100], Loss: %.4f' % (epoch+1, loss.item()))


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
    t_lo, t_hi, f_lo, f_hi = SFT.extent(N, center_bins=True)
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
    maxLength = 0
    # max seconds 
    maxLength = 16000 * 11
    for i in range(len(data)):
        # pad or truncate data to 11 seconds
        tensor = torch.tensor(data[i]['audio']['array']).type(torch.float32)
        if len(tensor) > maxLength:
            tensor = tensor[:maxLength]
        elif maxLength - len(tensor) > 0:
            tensor = torch.nn.functional.pad(tensor, (0, maxLength - len(tensor)))
        transformed = transformAudio(tensor, data[i]['audio']['sampling_rate'])
        arrays.append(torch.tensor(transformed).type(torch.float32))
        # make label 0 for Korean, 1 for English
        if data[i]['language'] == 'Korean':
            labels.append(torch.tensor([0]))
        elif data[i]['language'] == 'English':
            labels.append(torch.tensor([1]))
        else:
            raise ValueError('Invalid language')            
       
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
    
main()

