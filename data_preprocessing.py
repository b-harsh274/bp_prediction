import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os

class DataPreprocessor:
    def __init__(self):
        self.seq_len = None

    def read_ecg_data(self, file_path):
        df = np.loadtxt(file_path) 
        if self.seq_len is None:
            self.seq_len = df.shape[1]
        
        return df

    def ecg_data_preprocess(self, signal_path, label_path):
        print('[INFO] Reading input file')
        if os.path.isfile(signal_path) and os.path.isfile(label_path):
            data = self.read_ecg_data(signal_path)
            rdata = self.read_ecg_data(label_path)
        else:
            print('Check input file path, skipping execution')
            return 0, 0

        print('[INFO] Data Loading Complete. Preparing Data')

        # Normalizing data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        x = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
        y = rdata.reshape(rdata.shape[0], rdata.shape[1], 1)

        return x, y
