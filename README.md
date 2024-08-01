# ECG & PPG Feature Extraction Using Time Series UNet Model

## Overview

This repository contains the implementation of ECG and PPG feature extraction using a Time Series UNet model. This code is part of my summer internship at Sensesemi, where we developed methods for cuffless blood pressure prediction from ECG and PPG signals.

## Project Structure

- `data_preprocessor.py`: Script for reading and preprocessing ECG data.
- `model.py`: Script for creating the Time Series UNet model.
- `trainer.py`: Script for training the model with the preprocessed data.
- `main.py`: Script to execute the model training.
- `data/ecg_sig_1.txt`: Sample input file containing ECG signals.
- `data/ecg_labels_.txt`: Sample input file containing ECG labels.
- `data/ppg_sig_1.txt`: Sample input file containing PPG signals.
- `data/ppg_labels_.txt`: Sample input file containing PPG labels.


## Additional Information

- A Kaggle notebook containing the code for other methods developed during the internship is available [here](https://www.kaggle.com/code/bharsh2/bp-calculation).
- The complete dataset used for training is uploaded on Kaggle [here](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset).
- Note: The data used for training this model had to be manually labeled, resulting in a small dataset of only 15 samples.

## Report

For detailed information about the project and the methods developed, please refer to the internship report uploaded in this repository.




