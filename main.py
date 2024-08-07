import os
from train import Trainer

def main():
    signal = 'path/to/signal'
    label = 'path/to/labels'
    output_path = 'path/to/output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trainer = Trainer(output_path)
    trainer.train_ecg(signal, label)

if __name__ == "__main__":
    main()
