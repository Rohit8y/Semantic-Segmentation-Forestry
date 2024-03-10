import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


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
        img_path = self.dataList[idx]
        image = Image.open(img_path)
        imageArray = np.array(image, np.uint8)
        RGBImage = imageArray[:, :, 0:3]

        # Convert back from numpy to Image
        RGBImage = Image.fromarray(RGBImage)

        mask_path = self.dict[img_path]
        mask = Image.open(mask_path)
        resize = transforms.Resize(size=(512, 512))
        convert_tensor = transforms.ToTensor()

        # visualize( original_image = image, RGB_image = RGBImage, ground_truth_mask = mask)
        # Convert back from numpy to mask image
        mask = Image.fromarray(np.asarray(mask) * 255)

        if self.transform:
            image = self.transform(RGBImage)
            mask = convert_tensor(mask)
        else:
            image, mask = self.transformData(RGBImage, mask)

        return image, mask

    def transformData(self, image, mask):
        # Resize
        # resize = transforms.Resize(size=(512, 512))
        # image = resize(image)
        # mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(1024, 1024))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random Grayscale
        # if random.random() > 0.5:
        #    image = transforms.Grayscale()(image)

        # Random color jitter
        # if random.random() > 0.5:
        #   jitter = transforms.ColorJitter(brightness=.5, hue=.3)
        #   image = jitter(image)

        # Random sharpness
        # if random.random() > 0.5:
        #   sharpness_adjuster = transforms.RandomAdjustSharpness(sharpness_factor=2)
        #   image = sharpness_adjuster(image)

        # Random contrast
        # if random.random() > 0.5:
        # autocontraster = transforms.RandomAutocontrast()
        # image = autocontraster(image)

        # Transform to tensor
        image = TF.to_tensor(image)

        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        image = norm(image)

        mask = TF.to_tensor(mask)
        return image, mask
