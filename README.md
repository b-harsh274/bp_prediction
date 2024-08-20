# ECG & PPG Feature Extraction Using Time Series UNet Model

## Overview

This repository contains the implementation of ECG and PPG feature extraction using a Time Series UNet model. This code is part of my project, where we developed methods for cuffless blood pressure prediction from ECG and PPG signals.

## Project Structure

- `data_preprocessor.py`: Script for reading and preprocessing ECG data.
- `model.py`: Script for creating the Time Series UNet model.
- `train.py`: Script for training the model with the preprocessed data.
- `main.py`: Script to execute the model training.
- `data`: Directory with sample input files containing ECG & PPG data.

## Requirements

Ensure you have Python installed, preferably version 3.8 or above. Install the necessary packages using pip:

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Prepare Your Data

- Place your ECG signal data and corresponding labels in separate files.
- Ensure the files are formatted correctly as required by the data_preprocessor.py script.

### 2. Set Up Your File Paths

In the main.py script, modify the following variables to point to your data:

```python
signal_path = 'path/to/signal'
label_path = 'path/to/labels'
output_path = 'path/to/output'
```
### 3. Run the Training Script

Execute the main.py script to start training the model:

```bash
python main.py
```

The model will be trained using the data specified, and the trained model's weights will be saved in the output_path directory.

### 4. Monitor Training

- During training, the model's progress will be printed to the console, including loss values and validation metrics.
- The model checkpoint with the best validation loss will be saved automatically.

## Additional Information

- A Kaggle notebook containing the code for other methods developed during the project is available [here](https://www.kaggle.com/code/bharsh2/bp-calculation).
- The complete dataset used for training is uploaded on Kaggle [here](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset).
- Note: The data used for training this model had to be manually labeled, resulting in a small dataset of only 15 samples.




