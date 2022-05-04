# All measurement values in SI units

import pyaudio
import numpy as np
import math
import json
import datetime
import struct
import matplotlib.pyplot as plt

Sound_Speed = 343
Mic_diag_dist = 0.08127
Mic_side_dist = 0.0578
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 4
RESPEAKER_INDEX = 2
CHUNK = 1024
RECORD_SECONDS = 5
FORMAT = pyaudio.paFloat32

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

p = pyaudio.PyAudio()

def Get_Raw_Audio():
    
    
    Data_arrays = {}

    stream = p.open(
                rate=RESPEAKER_RATE,
                format=FORMAT,
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,)

    print("Recording Started")
    b1 = []
    b2 = []
    b3 = []
    b4 = []

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        
        data = stream.read(CHUNK)
        
        # extract audio data from 4 mic channels
        
        #b = np.fromstring(data,dtype=np.float32)[0::4]
        #b1 = struct.unpack(str(CHUNK*4)+'f', data)[0::4]
        b1 = np.append(b1, struct.unpack(str(CHUNK*4)+'f', data)[0::4])
        
        #bm2 = np.fromstring(data,dtype=np.float32)[1::4]
        b2 = np.append(b2, struct.unpack(str(CHUNK*4)+'f', data)[1::4])
        
        #bm3 = np.fromstring(data,dtype=np.float32)[2::4]
        b3 = np.append(b3, struct.unpack(str(CHUNK*4)+'f', data)[2::4])
        
        #bm4 = np.fromstring(data,dtype=np.float32)[3::4]
        b4 = np.append(b4, struct.unpack(str(CHUNK*4)+'f', data)[3::4])
        
    Data_arrays = {0: b3, 1: b4, 2: b2, 3: b1}
    

    print("Recording Ended")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    
    return Data_arrays
    
    
def main():
    
    Data_sample = Get_Raw_Audio()
    print(Data_sample.keys())
    print(Data_sample.values())
    
    x = np.arange(0, 79872)
    
    plt.plot(x, Data_sample[0], label='0')
    plt.plot(x, Data_sample[1], label='1')
    plt.plot(x, Data_sample[2], label='2')
    plt.plot(x, Data_sample[3], label='3')
    plt.legend()
    plt.show()
    
    file1 = open('/home/pi/Documents/data_45_180.json', 'w')

    json.dump(Data_sample, file1, cls=NumpyEncoder)
    file1.close()


if __name__ == "__main__":
    main()

# "arecord -Dac108 -f S32_LE -r 16000 -c 4 hello.wav"  Record sound on cmd line
# "Input Device id  2  -  seeed-4mic-voicecard: bcm2835-i2s-ac10x-codec0 ac10x-codec0-0 (hw:3,0)"
#current_time = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

