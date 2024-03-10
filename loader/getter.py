from torchvision import transforms
from torch.utils.data import DataLoader

from loader.forestry_dataset import GoettingenDataset

transformMyData = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_train_loader(args, shuffle=True):
    trainData = GoettingenDataset(args.data_path, train=True, transform=transformMyData)
    train_dataloader = DataLoader(trainData, batch_size=args.batch_size, shuffle=shuffle)
    return train_dataloader


def get_test_loader(args, shuffle=False):
    testData = GoettingenDataset(args.data_path, train=False, transform=transformMyData)
    test_dataloader = DataLoader(testData, batch_size=1, shuffle=shuffle)
    return test_dataloader
