import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import logging
from easyfl.datasets.simulation import data_simulation, split_s_and_u
from easyfl.datasets.utils.base_dataset import BaseDataset, TINY_IMAGENET
from easyfl.datasets.utils.util import save_dict
from easyfl.datasets.utils.download import download_url, extract_archive, download_from_google_drive


logger = logging.getLogger(__name__)
import h5py
from torch.utils.data import Dataset, DataLoader
import collections
import numpy as np
import sys
import os
from PIL import Image



class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        #self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images=collections.defaultdict(list)
        #self.images = {}
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                            self.images[self.class_to_tgt_idx[tgt]].append(path)
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                            self.images[self.class_to_tgt_idx[self.val_img_to_class[fname]]].append(path)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def return_data(self):
        return self.images


    def save_dataH5(self):
        self._make_dataset(self.Train)
        images_nplist=[]
        tgt_nplist=[]
        for tgt,path_lists in self.images.items():
            images = []
            for img_path in path_lists:
                with open(img_path, 'rb') as f:
                    sample = Image.open(img_path)
                    sample = sample.convert('RGB')

                    newsize = (32, 32)
                    sample = sample.resize(newsize, Image.ANTIALIAS)
                    sample=np.array(sample)[None,:,:,:]
                    images.append(sample)
            num_imgs=len(images)
            images_np=np.concatenate(images)
            images_tgt=np.full_like(np.ones((num_imgs,)),tgt,dtype=np.int32)

            images_nplist.append(images_np)
            tgt_nplist.append(images_tgt)
        data_x=np.concatenate(images_nplist)
        data_y=np.concatenate(tgt_nplist)

        if self.Train:
            path=os.path.join(self.root_dir,'train.h5')
        else:
            path=os.path.join(self.root_dir,'test.h5')
        save_h5(path,data_x,data_y)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]

        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
            newsize = (32, 32)
            sample = sample.resize(newsize, Image.ANTIALIAS)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, int(tgt)

class Tiny_imagenet(BaseDataset):
    def __init__(self,
                 root,
                 fraction,
                 split_type,
                 user,
                 iid_user_fraction=0.1,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=80,
                 num_of_client=100,
                 class_per_client=2,
                 setting_folder=None,
                 seed=-1,
                 weights=None,
                 alpha=0.5,
                 ssl_senario='server',
                 num_labels_per_class=5,
                 is_ssl=False):
        super(Tiny_imagenet, self).__init__(root,
                                       TINY_IMAGENET,
                                       fraction,
                                       split_type,
                                       user,
                                       iid_user_fraction,
                                       train_test_split,
                                       minsample,
                                       num_class,
                                       num_of_client,
                                       class_per_client,
                                       setting_folder,
                                       seed)

        self.train_data, self.test_data = {}, {}
        self.split_type = split_type
        self.num_of_client = num_of_client
        self.weights = weights
        self.alpha = alpha
        self.min_size = minsample
        self.class_per_client = class_per_client
        self.ssl_senario = ssl_senario
        self.num_labels_per_class = num_labels_per_class
        self.is_ssl = is_ssl
        self.size = (-1, 32, 32, 3)
        self.num_classes = 200


        self.packaged_data_files = {
            "tiny-imagenet-200.zip": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
            }


    def download_packaged_dataset_and_extract(self, filename):
        # file_path = download_url(self.packaged_data_files[filename], self.base_folder)
        # extract_archive(file_path, remove_finished=True)
        pass

    def extract_data(self, dataset, mode='train'):
        for j in range(self.num_classes):
            data = np.zeros((1, 32, 32, 3))
            label = np.zeros(1)
            sub_num = int(len(dataset.images)/self.num_classes)
            for i in range(sub_num):
                item, tgt = dataset.__getitem__(i + j * sub_num)
                item = np.array(item)[None, :, :, :]
                data = np.concatenate((data, item))
                label = np.concatenate((label, [tgt]))

            #aviod perform concatenation operation between high-dimension arrays
            if j > 0:
                tmp_dat = np.load(mode+"_data.npy")
                tmp_tar = np.load(mode+"_target.npy")
                tmp_dat = np.concatenate((tmp_dat, data[1:]))
                tmp_tar = np.concatenate((tmp_tar, label[1:]))
                np.save(mode+"_data.npy", tmp_dat)
                np.save(mode+"_target.npy", tmp_tar)

            else:
                np.save(mode+"_data.npy", np.array(data[1:]))
                np.save(mode+"_target.npy", np.array(label[1:]))

        total_data = np.load(mode+"_data.npy")
        total_label = np.load(mode+"_target.npy")

        os.remove(mode+"_data.npy")
        os.remove(mode+"_target.npy")

        return total_data.astype(np.float64), total_label.astype(np.int32)


    def download_raw_file_and_extract(self):
        #dataset_dir = os.path.join(self.base_folder, TINY_IMAGENET)
        data_path="/mnt/lustre/share_data/lishuaicheng/fssl_data/tiny-imagenet-200"
        if 'test.h5' not in os.listdir(data_path):
            dataset_train = TinyImageNet(data_path, train=True)
            dataset_test = TinyImageNet(data_path, train=False)
            dataset_train.save_dataH5()
            print("finied data_train")
            dataset_test.save_dataH5()
            print("finied data_test")
        train_path=os.path.join(data_path,'train.h5')
        test_path = os.path.join(data_path, 'test.h5')
        print("processing h5 file datas")
        train_x,train_y=read_h5(train_path,'x','y')
        test_x,test_y=read_h5(test_path,'x','y')

        # train_data, train_label = self.extract_data(dataset_train, 'train')
        # print('train data finish')
        # test_data, test_label = self.extract_data(dataset_test, 'test')
        # print('test data finish')


        self.train_data = {
            'x': train_x,
            'y': train_y.astype(np.int64).tolist()
        }
        self.test_data = {
            'x': test_x,
            'y': test_y.tolist()
        }

    def preprocess(self):
        if self.is_ssl:
            s_train_data_path = os.path.join(self.data_folder, "train_s_{}_{}".format(self.ssl_senario, self.num_labels_per_class))
            u_train_data_path = os.path.join(self.data_folder, "train_u_{}_{}".format(self.ssl_senario, self.num_labels_per_class))
        else:
            train_data_path = os.path.join(self.data_folder, "train")

        test_data_path = os.path.join(self.data_folder, "test")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        if self.weights is None and ((not self.is_ssl and os.path.exists(train_data_path)) or (self.is_ssl and (os.path.exists(u_train_data_path) or os.path.exists(s_train_data_path)))):
            return

        if self.is_ssl and self.ssl_senario != "client_local":
            #ssl_senario, num_labels_per_class, num_of_client
            self.s_train, self.u_train= split_s_and_u(self.train_data['x'], self.train_data['y'], self.ssl_senario, self.num_labels_per_class, self.num_of_client)


        logger.info("Start Tiny ImageNet data simulation")
        if self.is_ssl:
            if self.ssl_senario != 'server':
                ssl_s_split_type = 'iid'
                _, s_train_data = data_simulation(self.s_train['x'],
                                                  self.s_train['y'].tolist(),
                                                  self.num_of_client,
                                                  ssl_s_split_type,
                                                  self.weights,
                                                  self.alpha,
                                                  self.min_size,
                                                  self.class_per_client)
            else:
                s_train_data = {
                    'x': self.s_train['x'],
                    'y': self.s_train['y'].tolist()
                }
            print("Finish labeled data")
            _, u_train_data = data_simulation(self.u_train['x'],
                                              self.u_train['y'].tolist(),
                                              self.num_of_client,
                                              self.split_type,
                                              self.weights,
                                              self.alpha,
                                              self.min_size,
                                              self.class_per_client)


        else:
            _, train_data = data_simulation(self.train_data['x'],
                                            self.train_data['y'],
                                            self.num_of_client,
                                            self.split_type,
                                            self.weights,
                                            self.alpha,
                                            self.min_size,
                                            self.class_per_client)
        logger.info("Complete Tiny ImageNet data simulation")



        if self.is_ssl:
            save_dict(s_train_data, s_train_data_path)
            save_dict(u_train_data, u_train_data_path)

        else:
            save_dict(train_data, train_data_path)

        if not os.path.exists(test_data_path):
            save_dict(self.test_data, test_data_path)

    def convert_data_to_json(self):
        pass

def save_h5(path,data_x,data_y):
    f=h5py.File(path,'w')
    f.create_dataset("x",data=data_x)
    f.create_dataset("y", data=data_y)
    f.close

def read_h5(path,key_data,key_label):
    x,y=None,None
    f=h5py.File(path,'r')
    x=f[key_data][:]
    y=f[key_label][:]
    f.close()
    return x,y

if __name__=='__main__':
    data=TinyImageNet('/mnt/lustre/share_data/lishuaicheng/fssl_data/tiny-imagenet-200')
    data._make_dataset(True)