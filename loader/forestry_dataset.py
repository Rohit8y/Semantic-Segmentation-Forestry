import os

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import io


class GoettingenDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (callable, optional): Option argument to define if it is a training dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dict = {}
        self.dataList = []
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.convert_tensor = transforms.ToTensor()

        # For train data loader
        if train:
            mypath = root_dir + "train/tiles/"
            fileListData = sorted(filter(lambda x: os.path.isfile(os.path.join(mypath, x)),
                                         os.listdir(mypath)))

            mypath2 = root_dir + "train/masks/"
            fileListLabel = sorted(filter(lambda x: os.path.isfile(os.path.join(mypath2, x)),
                                          os.listdir(mypath2)))

            for index, file in enumerate(fileListData):
                if file.endswith('.tif'):
                    self.dataList.append(mypath + file)
                    self.dict[mypath + file] = mypath2 + fileListLabel[index]

                    # For test data loader
        else:
            mypath = root_dir + "test/tiles/"
            fileListData = sorted(filter(lambda x: os.path.isfile(os.path.join(mypath, x)),
                                         os.listdir(mypath)))

            mypath2 = root_dir + "test/masks/"
            fileListLabel = sorted(filter(lambda x: os.path.isfile(os.path.join(mypath2, x)),
                                          os.listdir(mypath2)))

            for index, file in enumerate(fileListData):
                if file.endswith('.tif'):
                    self.dataList.append(mypath + file)
                    self.dict[mypath + file] = mypath2 + fileListLabel[index]

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.dataList[idx]
        image = io.imread(img_name)
        numpyImage = np.asarray(image)
        # numpyImage = numpyImage[:,:,0:3]

        trueLabel = self.dict[img_name]
        trueLabel = io.imread(trueLabel)
        numpydata = np.asarray(trueLabel)

        trueLabel = self.convert_tensor(trueLabel)

        if self.transform:
            image = self.transform(numpyImage)

        return image, trueLabel
