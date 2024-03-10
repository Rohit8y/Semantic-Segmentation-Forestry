import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch MocoV2 pre-training')
parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: resnet 18|34|50|101|152')
parser.add_argument('--epochs', metavar='EPOCHS', default=100, type=int,
                    help='Number of epochs to train our network for')
parser.add_argument('--lr', metavar='LR', default=0.01, type=float,
                    help='Learning rate for training the model')
parser.add_argument('--batch_size', metavar='BS', default=16, type=int,
                    help='mini batch size for training')
parser.add_argument('--optimizer', metavar='OPT', default='sgd', type=str,
                    help='optimizer for updating the weights of the model: [sgd,adam]')
parser.add_argument('--momentum', metavar='M', default=0.9, type=float,
                    help='momentum used in case of SGD optimizer')

if __name__ == '__main__':
    print("Parsing arguments")
    args = parser.parse_args()
