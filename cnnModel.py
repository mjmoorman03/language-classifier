from datasets import load_dataset, concatenate_datasets, Audio
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, spectrogram, ShortTimeFFT
import json
import scipy
import audioInputUserChoice as audioInput
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

BATCH_SIZE = 16
NUM_EPOCHS = 10
MAX_LENGTH = 16000 * 11
LEARNING_RATE = 0.001
NUM_FILTERS = 16
NUM_SECOND_FILTERS = 32
NUM_THIRD_FILTERS = 64
NUM_FOURTH_FILTERS = 128
SAMPLING_RATE = 16000
NPERSEG = 256
NOVERLAP = 128
NUM_INCHANNELS = 2
LANGUAGES = ['Korean', 'English', 'Spanish']

class Model(torch.nn.Module):
    def __init__(self, numLanguages):
        super(Model, self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels=NUM_INCHANNELS, out_channels=NUM_FILTERS, kernel_size=3, stride=1, padding=1)
        self.cnn2 = torch.nn.Conv2d(in_channels=NUM_FILTERS, out_channels=NUM_SECOND_FILTERS, kernel_size=3, stride=1, padding=1)
        self.cnn3 = torch.nn.Conv2d(in_channels=NUM_SECOND_FILTERS, out_channels=NUM_THIRD_FILTERS, kernel_size=3, stride=1, padding=1)
        self.cnn4 = torch.nn.Conv2d(in_channels=NUM_THIRD_FILTERS, out_channels=NUM_FOURTH_FILTERS, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(688 * NUM_FOURTH_FILTERS, numLanguages)
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
    

    def trainFunction(self, trainLoader):
        losses = []
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(NUM_EPOCHS):
            for _, (trainData, trainLabels) in enumerate(trainLoader, 0):
                trainData, trainLabels = trainData.to(device), trainLabels.to(device)
                optimizer.zero_grad()
                outputs = self(trainData)
                trainLabels = trainLabels.squeeze(1)
                # unclear on shape of outputs and trainLabels, if we need to take argmax of output first
                loss = criterion(outputs, trainLabels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            print(f'Epoch [%d/{NUM_EPOCHS}], Loss: %.5f' % (epoch+1, loss.item()))
        return losses

        
    def test(self, testLoader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testLoader:
                testInputs, testLabels = data
                testInputs, testLabels = testInputs.to(device), testLabels.to(device)
                outputs = self(testInputs)
                predicted = torch.argmax(outputs, 1)
                testLabels = testLabels.squeeze(1)
                total += testLabels.size(0)
                correct += (predicted == testLabels).sum().item()
        print(f'Accuracy of the network on the test data: {100 * correct/total}')
        return 100 * correct / total
    
    
    def spectrogramFeatures(self, x):
        x = x.to(device)
        x = self.cnn1(x)
        plotSpectrogram(x[0][3].detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        x = self.batchNorm1(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.cnn2(x)
        plotSpectrogram(x[0][3].detach().cpu().numpy())
        spectrogramToAudio(x[0][3].detach().cpu().numpy(), factor=2)
        x = torch.nn.functional.relu(x)
        x = self.batchNorm2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.cnn3(x)
        plotSpectrogram(x[0][3].detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        x = self.batchNorm3(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.cnn4(x)
        plotSpectrogram(x[0][3].detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        x = self.batchNorm4(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)


def transformAudio(audio, samplingRate):
    # before, nperseg was different from this, but they are supposed to be the same. perhaps this will help
    win = ('gaussian', NPERSEG)
    SFT = ShortTimeFFT.from_window(win, samplingRate, NPERSEG, NOVERLAP, fft_mode='onesided',
                               scale_to='magnitude', phase_shift=None)
    transformed = SFT.stft(audio)
    return transformed


def plotSpectrogram(image):
    fig1, axx = plt.subplots(1, 1, sharex='all', sharey='all',
                            figsize=(6., 5.))  # enlarge figure a bit
    axx.set_title(r"ShortTimeFFT produces $%d\times%d$ points" % image.shape)
    axx.set_xlabel(r"Time $t$ in seconds")
    # Calculate extent of plot with centered bins since
    # imshow does not interpolate by default:
    extent1 = [0, 11, 0, 8000]
    kw = dict(origin='lower', aspect='auto', cmap='viridis')
    im1b = axx.imshow(abs(image), extent=extent1, **kw)
    fig1.colorbar(im1b, ax=axx, label="Magnitude $|S_z(t, f)|$")
    _ = fig1.supylabel(r"Frequency $f$ in Hertz", x=0.08, y=0.5, fontsize='medium')
    plt.show()


def plotSpectrogramFromFFT(audio, samplingRate): 
    win = ('gaussian', NPERSEG)
    SFT = ShortTimeFFT.from_window(win, samplingRate, NPERSEG, NOVERLAP, fft_mode='onesided',
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


def plotAudio(audio, samplingRate):
    plt.plot(audio)
    plt.xticks(np.arange(0, len(audio), samplingRate), np.arange(0, len(audio)/samplingRate, 1))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def spectrogramToAudio(spectrogram, factor=1):
    # inverse short time fourier transform
    win = ('gaussian', NPERSEG/factor)
    _, audio_signal = istft(spectrogram, fs=SAMPLING_RATE, nperseg=(NPERSEG-2)/factor, noverlap=NOVERLAP/factor, input_onesided=True, window=win)
    audio_signal = np.where(np.abs(audio_signal) >= np.max(audio_signal)/100, audio_signal, 0)
    plotAudio(abs(audio_signal), SAMPLING_RATE)
    scipy.io.wavfile.write('reconstructed_audio.wav', SAMPLING_RATE, audio_signal.astype(np.float32))


def createDataloader(batch_size=BATCH_SIZE, *data):
    arrays = []
    labels = []
    SAMPLINGRATE = 16000
    LOCALES = ['ko', 'en', 'es']
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        TimeMask( min_band_part=0.1, max_band_part=0.15, fade=True, p=0.5,),
    ])
    # max seconds 
    for i in range(len(data[0])):
        # pad or truncate data to 11 seconds
        audio = data[0][i]['audio']['array']
        audio = augment(samples=audio, sample_rate=16000)
        tensor = torch.tensor(audio).type(torch.float32)
        if len(tensor) > MAX_LENGTH:
            tensor = tensor[:MAX_LENGTH]
        elif MAX_LENGTH - len(tensor) > 0:
            tensor = torch.nn.functional.pad(tensor, (0, MAX_LENGTH - len(tensor)))
        transformed = transformAudio(tensor, data[0][i]['audio']['sampling_rate'])
        transformedReal = np.real(transformed)
        transformedImag = np.imag(transformed)
        transformedReal = torch.tensor(abs(transformedReal)).type(torch.float32)
        transformedImag = torch.tensor(abs(transformedImag)).type(torch.float32)
        arrays.append(torch.stack((transformedReal, transformedImag), 0))
        # labels for languages to indices
        if data[0][i]['language'] in LANGUAGES:
            labels.append(torch.tensor([LANGUAGES.index(data[0][i]['language'])]))
        else:
            raise ValueError('Invalid language')    
    
    if (data[1]):
        for i in range(len(data[1])):
            # pad or truncate data to 11 seconds
            audio = data[1][i]['audio']['array']
            audio = augment(samples=audio, sample_rate=16000)
            tensor = torch.tensor(audio).type(torch.float32)
            if len(tensor) > MAX_LENGTH:
                tensor = tensor[:MAX_LENGTH]
            elif MAX_LENGTH - len(tensor) > 0:
                tensor = torch.nn.functional.pad(tensor, (0, MAX_LENGTH - len(tensor)))
            transformed = transformAudio(tensor, SAMPLINGRATE)
            transformedReal = np.real(transformed)
            transformedImag = np.imag(transformed)
            transformedReal = torch.tensor(abs(transformedReal)).type(torch.float32)
            transformedImag = torch.tensor(abs(transformedImag)).type(torch.float32)
            arrays.append(torch.stack((transformedReal, transformedImag), 0))
            # labels for languages to indices
            if data[1][i]['locale'] in LOCALES:
                labels.append(torch.tensor([LOCALES.index(data[1][i]['locale'])]))
            else:
                raise ValueError('Invalid language')

    xTrain, xTest, yTrain, yTest = train_test_split(arrays, labels, test_size=0.15, random_state=None)
    xTrain = torch.stack(xTrain)
    xTest = torch.stack(xTest)
    yTrain = torch.stack(yTrain)
    yTest = torch.stack(yTest)
    trainData = torch.utils.data.TensorDataset(xTrain, yTrain) # type: ignore
    testData = torch.utils.data.TensorDataset(xTest, yTest) # type: ignore
        
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True) # type: ignore
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=False) # type: ignore
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
    transformed = transformAudio(audio, samplingRate)
    transformedReal = np.real(transformed)
    transformedImag = np.imag(transformed)
    transformedReal = torch.tensor(abs(transformedReal)).type(torch.float32)
    transformedImag = torch.tensor(abs(transformedImag)).type(torch.float32)
    plotSpectrogram(transformedReal)
    plotSpectrogram(transformedImag)
    transformedReal = transformedReal.to(device)
    transformedImag = transformedImag.to(device)
    transformed = torch.stack((transformedReal, transformedImag), 0)
    transformed = transformed.unsqueeze(0)
    outputs = model(transformed)
    print(outputs)
    predicted = torch.argmax(outputs, 1)
    return LANGUAGES[predicted]


def classifyMicrophoneInput(model):
    audio = audioInput.audioInputUserChoice()
    plotAudio(audio, SAMPLING_RATE)
    return evaluateAudio(model, audio, SAMPLING_RATE)


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
    fleurs_korean = load_dataset('google/fleurs', "ko_kr", split='train[:80%]') # type: ignore
    fleurs_english = load_dataset('google/fleurs', "en_us", split='train[:80%]') 
    fleurs_spanish = load_dataset('google/fleurs', "es_419", split='train[:80%]')
    # fleurs_vietnamese = load_dataset('google/fleurs', "vi_vn", split='train', trust_remote_code=True)

    data = concatenate_datasets([fleurs_korean, fleurs_english, fleurs_spanish]) # type: ignore

    # This entire block can be commented out and common_voice_data deleted when we want to use it or not use it
    # Also for sake of RAM I have lowered training data amount of fleurs when I use this
    # Don't know if u need to but yeah
    common_voice_korean = load_dataset('mozilla-foundation/common_voice_13_0', "ko", split='train')
    common_voice_korean2 = load_dataset('mozilla-foundation/common_voice_13_0', "ko", split='test')
    common_voice_korean3 = load_dataset('mozilla-foundation/common_voice_13_0', "ko", split='validation')
    common_voice_english = load_dataset('mozilla-foundation/common_voice_13_0', "en", split='test[:10%]')
    common_voice_spanish = load_dataset('mozilla-foundation/common_voice_13_0', "es", split='test[:10%]')
    common_voice_korean = common_voice_korean.cast_column("audio", Audio(sampling_rate=16000))
    common_voice_korean2 = common_voice_korean2.cast_column("audio", Audio(sampling_rate=16000))
    common_voice_korean3 = common_voice_korean3.cast_column("audio", Audio(sampling_rate=16000))
    common_voice_english = common_voice_english.cast_column("audio", Audio(sampling_rate=16000))
    common_voice_spanish = common_voice_spanish.cast_column("audio", Audio(sampling_rate=16000))
    common_voice_data = concatenate_datasets([common_voice_korean, common_voice_korean2, common_voice_korean3, common_voice_english])


    trainLoader, testLoader = createDataloader(BATCH_SIZE, data, common_voice_data)

    model = Model(len(LANGUAGES))
    model.to(device)
    model.trainFunction(trainLoader)
    accuracy = model.test(testLoader)
    saveModelAccuracy(model, accuracy)
    saveResults(model, accuracy)


def featurizeInput(model, audio, samplingRate):
    transformed = transformAudio(audio, samplingRate)
    transformedReal = torch.tensor(abs(np.real(transformed))).type(torch.float32)
    transformedImag = torch.tensor(abs(np.imag(transformed))).type(torch.float32)
    plotSpectrogram(transformedReal)
    transformedReal = transformedReal.to(device)
    transformedImag = transformedImag.to(device)
    transformed = torch.stack((transformedReal, transformedImag), 0)
    transformed = transformed.unsqueeze(0)
    return model.spectrogramFeatures(transformed)


def testAudioConversion():
    audio = audioInput.audioInputUserChoice()
    plotAudio(audio, 16000)
    transformed = transformAudio(audio, SAMPLING_RATE)
    plotSpectrogram(transformed)
    spectrogramToAudio(transformed)


def beforeAfterGraph():
    audio = audioInput.audioInputUserChoice()
    fig, axs = plt.subplots(2)
    axs[0].plot(audio)
    axs[0].set_title('Before')
    transformed = transformAudio(audio, SAMPLING_RATE)
    spectrogramToAudio(transformed)
    plt.show()


def main():
    mainTrain()
    # model = loadModel()
    # model.to(device)
    # print(classifyMicrophoneInput(model))
    #audio = audioInput.audioInputUserChoice()
    #features = featurizeInput(model, audio, SAMPLING_RATE)
    #testModel()
    #beforeAfterGraph()
    #testAudioConversion()



if __name__ == '__main__':
    main()

