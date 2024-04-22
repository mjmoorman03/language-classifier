import pyaudio
import time
from threading import Thread
import torch
import numpy as np

# Need to brew install portaudio before pip install pyaudio

class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
thread_running = True
def audioInput():
    global thread_running

    CHANNELS = 1
    # RECORD_SECONDS = 10
    CHUNK = 1024
    RATE = 16000
    FORMAT = pyaudio.paFloat32
    maxLength = 16000 * 11

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
    frames = bytearray()

    while thread_running:
        data = stream.read(CHUNK)
        frames.extend(data)
    

    print('Recording finished\n')
    stream.stop_stream()
    stream.close()
    p.terminate()

    frames = np.frombuffer(frames, dtype=np.float32)
    tensor = torch.tensor(frames, dtype= torch.float32)
    tensor = abs(tensor)
    print(tensor)

    if len(tensor) > maxLength:
            tensor = tensor[:maxLength]
    elif maxLength - len(tensor) > 0:
        tensor = torch.nn.functional.pad(tensor, (0, maxLength - len(tensor)))

    return tensor

def wait_for_input():
    time.sleep(1)
    time.sleep(1)
    time.sleep(1)
    wait = input('Press enter to stop recording\n')

def audioInputUserChoice():
    global thread_running
    t1 = ThreadWithReturnValue(target=audioInput)
    t2 = Thread(target=wait_for_input)

    t1.start()
    t2.start()
    
    t2.join()
    thread_running = False
    return (t1.join())

if __name__ == '__main__':
    audioInputUserChoice()



    