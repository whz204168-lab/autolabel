import pickle
import numpy as np
import torch
from numpy import linalg
import torchvision.transforms as transforms

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


# %%

def get_cifar10(X_high, lblh, n_labeled,
                 transform_train=None, transform_val=None):
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(lblh, int(n_labeled / 11))
    train_labeled_dataset = CIFAR10_labeled(X_high, lblh, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = CIFAR10_unlabeled(X_high,lblh, train_unlabeled_idxs, train=True,
                                                transform=TransformTwice(transform_train))
    val_dataset = CIFAR10_labeled(X_high, lblh, val_idxs, train=True, transform=transform_val)
    test_dataset = CIFAR10_labeled(X_high, lblh, val_idxs, train=False, transform=transform_val)
    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(11):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-3000])
        val_idxs.extend(idxs[-3000:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def normalise(x):
    # x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    for i in range(x.shape[0]):
        x[i, 0, :, :] = x[i, 0, :, :] / linalg.norm(x[i, 0, :, :], 2)
        x[i, 1, :, :] = x[i, 1, :, :] / linalg.norm(x[i, 1, :, :], 2)
    # x *= 1.0/(255*std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

# 假设调制

def pad(x, border=4):
    return np.pad(x, [(0, 0), (0,0),(border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = output_size
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        w, h = x.shape[1:]
        new_h = self.output_size

        top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)

        x = x[:, :, top: top + new_h]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, ::-1, :]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.05
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class CIFAR10_labeled(object):

    def __init__(self, X, lblh, indexs=None, train=True,
                 transform=None, target_transform=None,):
        self.transform = transform
        self.target_transform = target_transform

        if indexs is not None:
            self.data = X[indexs]
            self.targets = np.array(lblh)[indexs]

        self.data = transpose(normalise(self.data))
        self.dataLen = self.data.shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.dataLen

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, X, lblh, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(X,lblh, indexs, train=train,
                                                transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for i in range(len(self.targets))])

