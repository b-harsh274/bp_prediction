import os
from train import Trainer

def main():
    signal_path = 'path/to/signal'
    label_path = 'path/to/labels'
    output_path = 'path/to/output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trainer = Trainer(output_path)
    trainer.train_ecg(signal_path, label_path)

if __name__ == "__main__":
    main()
