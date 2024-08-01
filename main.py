import os
from train import Trainer

def main():
    in_ecg_path = '/Users/harshbalgude/Desktop/hb/CufflessBP/combined/unet/ecg_sig_1.txt'
    in_wout_ecg_path = '/Users/harshbalgude/Desktop/hb/CufflessBP/combined/unet/ecg_labels_1.txt'
    output_path = '/Users/harshbalgude/Desktop/hb/CufflessBP/combined/unet/model_out'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trainer = Trainer(output_path)
    trainer.train_ecg(in_ecg_path, in_wout_ecg_path)

if __name__ == "__main__":
    main()
