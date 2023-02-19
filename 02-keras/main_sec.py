import argparse

import trainers as trainer_import
# noinspection PyUnresolvedReferences
from trainers import trainer_cnn


def main():
    trainer = trainer_cnn.TrainerCNN(
        epochs = 40, learning_rate = 0.001, val_split = 0.5, train_split = 0.5) # val_split
    trainer.train()

if __name__ == "__main__":
    main()
