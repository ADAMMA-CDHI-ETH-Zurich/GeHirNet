import os
import numpy as np
import soundfile as sf

def min_max_normalize_audio(input_file, output_file):

    data, samplerate = sf.read(input_file)
    
    # Check if the audio is mono or multi-channel
    if len(data.shape) > 1:  # Multi-channel audio
        normalized_data = []
        for channel in data.T:
            min_val = np.min(channel)
            max_val = np.max(channel)
            normalized_channel = (channel - min_val) / (max_val - min_val)
            normalized_data.append(normalized_channel)
        normalized_data = np.array(normalized_data).T
    else:  # Mono audio
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    
    # save
    sf.write(output_file, normalized_data, samplerate)
    print(f"Audio {input_file} normalized and save to {output_file}")

input_dir = r'path\to\input_audios'  
output_dir = r'path\to\output_audios' 

for file_name in os.listdir(input_dir):
    if file_name.endswith('.wav'):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        min_max_normalize_audio(input_path, output_path)
