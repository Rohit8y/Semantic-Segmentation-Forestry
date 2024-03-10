import argparse
import os.path

import torch
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

from loader.getter import get_train_loader, get_test_loader
from models.arch_config import DeepLabModel
from utils.early_stopping import EarlyStopping
from utils.metrics import getDiceLoss, getIoU
from utils.plots import generate_plots

parser = argparse.ArgumentParser(description='PyTorch MocoV2 pre-training')
parser.add_argument('--data_path', metavar='PATH', default='data/', type=str,
                    help='the path to the dataset generated')
parser.add_argument('--output_path', metavar='OUTPUT', default='output/', type=str,
                    help='output directory for all models and plots')
parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: resnet 18|34|50|101|152')
parser.add_argument('--epochs', metavar='EPOCHS', default=5, type=int,
                    help='Number of epochs to train our network for')
parser.add_argument('--lr', metavar='LR', default=0.00005, type=float,
                    help='Learning rate for training the model')
parser.add_argument('--wd', metavar='WD', default=0.001, type=float,
                    help='the weight decay value for decaying the weights of the model')
parser.add_argument('--batch_size', metavar='BS', default=2, type=int,
                    help='mini batch size for training')
parser.add_argument('--optimizer', metavar='OPT', default='adam', type=str,
                    help='optimizer for updating the weights of the model: [sgd,adam]')
parser.add_argument('--momentum', metavar='M', default=0.9, type=float,
                    help='momentum used in case of SGD optimizer')


def train(args, train_epoch, test_epoch, train_dataloader, test_dataloader):
    print("Training...")
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001, mode='min', model=model,
                                   output_path=args.output_path)
    train_logs_list, test_logs_list = [], []

    for i in range(0, args.epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        test_logs = test_epoch.run(test_dataloader)
        train_logs_list.append(train_logs)
        test_logs_list.append(test_logs)

        early_stopping(test_logs['dice_loss'], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print("Best IoU:", early_stopping.best_score)
    generate_plots(args, train_logs_list, test_logs_list)


if __name__ == '__main__':
    print("Parsing arguments...")
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Generate model
    obj = DeepLabModel(arch=args.arch)
    model = obj.get_model()
    print("Model obtained...")

    # Generate dataloaders
    train_loader = get_train_loader(args, shuffle=True)
    test_loader = get_test_loader(args, shuffle=False)
    print("Size of train loader:", len(train_loader))

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function
    lossFunction = getDiceLoss()

    # define metrics
    metrics = getIoU()

    # define optimizer
    optimizer = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(momentum=args.momentum, params=model.parameters(), lr=args.lr, weight_decay=args.wd)

    # define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # Generate train and test epoch functions
    train_epoch = TrainEpoch(model, loss=lossFunction, metrics=metrics, optimizer=optimizer, device=DEVICE,
                             verbose=True)
    test_epoch = ValidEpoch(model, loss=lossFunction, metrics=metrics, device=DEVICE, verbose=True)
    print("Train and test epoch function successfully loaded...")

    # Training block
    train(args, train_epoch, test_epoch, train_loader, test_loader)
