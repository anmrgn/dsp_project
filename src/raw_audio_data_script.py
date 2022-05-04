# All measurement values in SI units

import pyaudio
import numpy as np
import math
import json
import datetime
import struct

Sound_Speed = 343
Mic_diag_dist = 0.08127
Mic_side_dist = 0.0578
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 4
RESPEAKER_INDEX = 2
CHUNK = 1024
RECORD_SECONDS = 3
FORMAT = pyaudio.paFloat32


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

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        
        data = stream.read(CHUNK)
        
        # extract audio data from 4 mic channels
        
        #b = np.fromstring(data,dtype=np.float32)[0::4]
        b1 = struct.unpack(str(CHUNK*4)+'f', data)[0::4]
        
        #bm2 = np.fromstring(data,dtype=np.float32)[1::4]
        b2 = struct.unpack(str(CHUNK*4)+'f', data)[1::4]
        
        #bm3 = np.fromstring(data,dtype=np.float32)[2::4]
        b3 = struct.unpack(str(CHUNK*4)+'f', data)[2::4]
        
        #bm4 = np.fromstring(data,dtype=np.float32)[3::4]
        b4 = struct.unpack(str(CHUNK*4)+'f', data)[3::4]
        
    Data_arrays = {0: b1, 1: b2, 2: b3, 3: b4}
    

    print("Recording Ended")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    
    return Data_arrays
    
    
#def main():
    
    #Data_sample = Get_Raw_Audio()
    #print(Data_sample.keys())
    #print(Data_sample.values())
    
    
#if __name__ == "__main__":
    #main()

# "arecord -Dac108 -f S32_LE -r 16000 -c 4 hello.wav"  Record sound on cmd line
# "Input Device id  2  -  seeed-4mic-voicecard: bcm2835-i2s-ac10x-codec0 ac10x-codec0-0 (hw:3,0)"
#current_time = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
