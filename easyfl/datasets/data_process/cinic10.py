import numpy as np
import torch
import torchvision
from torchvision import transforms
from .randaugment import RandAugmentMC
from PIL import Image
class Cutout(object):
    """Cutout data augmentation is adopted from https://github.com/uoguelph-mlrg/Cutout"""

    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).

        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it. 
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

mean=(0.47889522, 0.47227842, 0.43047404)
std=(0.24205776, 0.23828046, 0.25874835)
transform_train = transforms.Compose([
    #torchvision.transforms.ToPILImage(mode='RGB'),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_train.transforms.append(Cutout())

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class TransformFixMatch(object):
    def __init__(self, mean=mean, std=std):
        self.weak = transforms.Compose([
            #torchvision.transforms.ToPILImage(mode='RGB'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            #torchvision.transforms.ToPILImage(mode='RGB'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


def load_imgarray_from_path(path_list):
    array_list=[]
    for path in path_list:
        img=pil_loader(path)
        array_list.append(np.asarray(img))

    return np.stack(array_list,axis=0)



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

#transform path to img
# def process_x(raw_x_batch):
#     path_list=raw_x_batch.tolist()
#     return load_imgarray_from_path(path_list)
