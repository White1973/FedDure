from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.images = images
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, label = self.images[index], self.labels[index]
        if self.transform_x is not None:
            img=Image.open(data)
            img=img.convert('RGB')

            data = self.transform_x(img)
        else:
            data = Image.open(data)
        if self.transform_y is not None:
            label = self.transform_y(label)
        return data, label


class TransformDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.data = images
        self.targets = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform_x:
            sample = self.transform_x(sample)
        if self.transform_y:
            target = self.transform_y(target)

        return sample, target


class TransformDataset_plus_idx(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.data = images
        self.targets = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform_x:
            sample = self.transform_x(sample)
        if self.transform_y:
            target = self.transform_y(target)

        return idx,sample, target



class TensorDataset_plus_idx(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return list(index,)+list(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)