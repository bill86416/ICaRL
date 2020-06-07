from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
import numpy as np
import torch
import os
from PIL import Image

class iCIFAR10(CIFAR10):
    def __init__(self, root,
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])
            self.trn_data = np.array(train_data)
            self.trn_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.tst_data = np.array(test_data)
            self.tst_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.trn_data[index], self.trn_labels[index]
        else:
            img, target = self.tst_data[index], self.tst_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.trn_data)
        else:
            return len(self.tst_data)

    def get_image_class(self, label):
        return self.trn_data[np.array(self.trn_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.trn_data = np.concatenate((self.trn_data, images), axis=0)
        self.trn_labels = self.trn_labels + labels

class ifashionmnist(FashionMNIST):
    def __init__(self, root,
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(ifashionmnist, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.train_data)):
                if self.train_labels[i] in classes:
                    train_data.append(self.data[i].data.numpy())
                    train_labels.append(int(self.targets[i].data.numpy()))
                    #print(self.targets[i].data.numpy())
                    #break
            #self.train_data = np.array(train_data)
            #self.train_labels = train_labels
            self.trn_data = np.array(train_data)
            self.trn_labels = train_labels


        else:
            test_data = []
            test_labels = []

            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    test_data.append(self.test_data[i].data.numpy())
                    test_labels.append(int(self.test_labels[i].data.numpy()))

            self.tst_data = np.array(test_data)
            self.tst_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.trn_data[index], self.trn_labels[index]
        else:
            img, target = self.tst_data[index], self.tst_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.trn_data)
        else:
            return len(self.tst_data)

    def get_image_class(self, label):
        return self.trn_data[np.array(self.trn_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.trn_data = np.concatenate((self.trn_data, images), axis=0)
        self.trn_labels = self.trn_labels + labels



class imnist(MNIST):
    def __init__(self, root,
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(imnist, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.train_data)):
                if self.train_labels[i] in classes:
                    train_data.append(self.data[i].data.numpy())
                    train_labels.append(int(self.targets[i].data.numpy()))
                    #print(self.targets[i].data.numpy())
                    #break
            #self.train_data = np.array(train_data)
            #self.train_labels = train_labels
            self.trn_data = np.array(train_data)
            self.trn_labels = train_labels


        else:
            test_data = []
            test_labels = []

            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    test_data.append(self.test_data[i].data.numpy())
                    test_labels.append(int(self.test_labels[i].data.numpy()))

            self.tst_data = np.array(test_data)
            self.tst_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.trn_data[index], self.trn_labels[index]
        else:
            img, target = self.tst_data[index], self.tst_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.trn_data)
        else:
            return len(self.tst_data)

    def get_image_class(self, label):
        return self.trn_data[np.array(self.trn_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.trn_data = np.concatenate((self.trn_data, images), axis=0)
        self.trn_labels = self.trn_labels + labels

class inotmnist():
    def __init__(self, root, 
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None):
        super(inotmnist, self)        # Select subset of classe

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            train_data = []
            train_labels = []
            data_file = 'train.pt'
            self.data, self.train_labels = torch.load(os.path.join(root, 'notmnist/'+data_file))
            for i in range(len(self.data)):
                if self.train_labels[i] in classes:
                    train_data.append(self.data[i].data.numpy())
                    train_labels.append(int(self.train_labels[i].data.numpy()))
            self.trn_data = np.array(train_data)
            self.trn_labels = train_labels
        else:
            test_data = []
            test_labels = []
            data_file = 'test.pt'
            self.test_data, self.test_labels = torch.load(os.path.join(root, 'notmnist/'+data_file))
 
            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    test_data.append(self.test_data[i].data.numpy())
                    test_labels.append(int(self.test_labels[i].data.numpy()))

            self.tst_data = np.array(test_data)
            self.tst_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.trn_data[index], self.trn_labels[index]
        else:
            img, target = self.tst_data[index], self.tst_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.trn_data)
        else:
            return len(self.tst_data)

    def get_image_class(self, label):
        return self.trn_data[np.array(self.trn_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.trn_data = np.concatenate((self.trn_data, images), axis=0)
        self.trn_labels = self.trn_labels + labels


class iCIFAR100(iCIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
