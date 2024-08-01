import os
from train import Trainer

def main():
    in_ecg_path = 'path/to/signal'
    in_wout_ecg_path = 'path/to/labels'
    output_path = 'path/to/output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trainer = Trainer(output_path)
    trainer.train_ecg(in_ecg_path, in_wout_ecg_path)

if __name__ == "__main__":
    main()
