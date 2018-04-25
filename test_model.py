from models import AlexNet

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['AlexNet', 'DenseNet'], required=True)
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist'], required=True)

    args = parser.parse_args()
