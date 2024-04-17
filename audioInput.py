import pyaudio
import time
import torch

# Need to brew install portaudio before pip install pyaudio
def audioInput():
    CHANNELS = 1
    RECORD_SECONDS = 10
    CHUNK = 1024
    RATE = 16000
    FORMAT = pyaudio.paInt16

    p = pyaudio.PyAudio()

    print('Recording in 3\n')
    time.sleep(1)
    print('2\n')
    time.sleep(1)
    print('1\n')
    time.sleep(1)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print('Recording\n')
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print('Recording finished\n')
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = torch.tensor(frames.type(torch.float32))
    return audio

if __name__ == '__main__':
    audioInput()