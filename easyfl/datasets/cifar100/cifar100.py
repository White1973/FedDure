import logging
import os
import pdb
import torchvision

from easyfl.datasets.simulation import data_simulation, split_s_and_u,data_simulation_localtest,split_s_and_u_dir
from easyfl.datasets.utils.base_dataset import BaseDataset, CIFAR100
from easyfl.datasets.utils.util import save_dict

logger = logging.getLogger(__name__)


class Cifar100_ori(BaseDataset):
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
        super(Cifar100_ori, self).__init__(root,
                                       CIFAR100,
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

    def download_packaged_dataset_and_extract(self, filename):
        pass

    def download_raw_file_and_extract(self):
        train_set = torchvision.datasets.CIFAR100(root=self.base_folder, train=True, download=False)
        test_set = torchvision.datasets.CIFAR100(root=self.base_folder, train=False, download=False)

        self.train_data = {
            'x': train_set.data,
            'y': train_set.targets
        }

        self.test_data = {
            'x': test_set.data,
            'y': test_set.targets
        }

    def preprocess(self):
        if self.is_ssl:
            s_train_data_path = os.path.join(self.data_folder, "train_s_{}".format(self.ssl_senario))
            u_train_data_path = os.path.join(self.data_folder, "train_u_{}".format(self.ssl_senario))
        else:
            train_data_path = os.path.join(self.data_folder, "train")
        test_data_path = os.path.join(self.data_folder, "test")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if self.weights is None and ((not self.is_ssl and os.path.exists(train_data_path)) or (
                self.is_ssl and (os.path.exists(u_train_data_path) or os.path.exists(s_train_data_path)))):
            return

        if self.is_ssl:
            # ssl_senario, num_labels_per_class, num_of_client
            self.s_train, self.u_train = split_s_and_u(self.train_data['x'], self.train_data['y'], self.ssl_senario,
                                                       self.num_labels_per_class, self.num_of_client)

        logger.info("Start CIFAR10 data simulation")
        if self.is_ssl:
            if self.ssl_senario != 'server':
                _, s_train_data = data_simulation(self.s_train['x'],
                                                  self.s_train['y'].tolist(),
                                                  self.num_of_client,
                                                  self.split_type,
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
        logger.info("Complete CIFAR100 data simulation")
        if self.is_ssl:
            save_dict(s_train_data, s_train_data_path)
            save_dict(u_train_data, u_train_data_path)
        else:
            save_dict(train_data, train_data_path)

        if not os.path.exists(test_data_path):
            save_dict(self.test_data, test_data_path)

    def convert_data_to_json(self):
        pass


class Cifar100(BaseDataset):
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
                 is_ssl=False,local_test=None,s_split_type=None):
        super(Cifar100, self).__init__(root,
                                       CIFAR100,
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
        self.local_test = local_test
        self.s_split_type = s_split_type
    def download_packaged_dataset_and_extract(self, filename):
        pass

    def download_raw_file_and_extract(self):
        train_set = torchvision.datasets.CIFAR100(root=self.base_folder, train=True, download=False)
        test_set = torchvision.datasets.CIFAR100(root=self.base_folder, train=False, download=False)

        self.train_data = {
            'x': train_set.data,
            'y': train_set.targets
        }

        self.test_data = {
            'x': test_set.data,
            'y': test_set.targets
        }

    def preprocess(self):
        if self.is_ssl:
            s_train_data_path = os.path.join(self.data_folder,
                                             "train_s_{}_{}".format(self.ssl_senario, self.num_labels_per_class))
            u_train_data_path = os.path.join(self.data_folder,
                                             "train_u_{}_{}".format(self.ssl_senario, self.num_labels_per_class))
        else:
            train_data_path = os.path.join(self.data_folder, "train")

        test_data_path = os.path.join(self.data_folder, "test")

        if self.local_test:
            local_test_data_path = os.path.join(self.data_folder, "local_test")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        if self.weights is None and ((not self.is_ssl and os.path.exists(train_data_path)) or (
                self.is_ssl and (os.path.exists(u_train_data_path) or os.path.exists(s_train_data_path)))):
            return

        # if self.is_ssl:
        #     #ssl_senario, num_labels_per_class, num_of_client
        #     self.s_train, self.u_train = split_s_and_u(self.train_data['x'], self.train_data['y'], self.ssl_senario, self.num_labels_per_class, self.num_of_client)

        logger.info("Start CIFAR100 data simulation")
        test_data = self.test_data

        if self.is_ssl and (self.s_split_type == 'dir' and (self.split_type == 'dir')):
            logger.info("Start CIFAR100 data dirichlet simulation")
            _, train_local_data, test_local_data = data_simulation_localtest(self.data_folder, self.train_data['x'],
                                                                             self.train_data['y'],
                                                                             self.test_data['x'],
                                                                             self.test_data['y'],
                                                                             self.num_of_client,
                                                                             self.split_type,
                                                                             self.weights,
                                                                             self.alpha,
                                                                             self.min_size,
                                                                             self.class_per_client, test=True)

            s_train_data, u_train_data = split_s_and_u_dir(train_local_data, self.ssl_senario,
                                                           self.num_labels_per_class, self.num_of_client)

        elif self.is_ssl:
            self.s_train, self.u_train = split_s_and_u(self.train_data['x'], self.train_data['y'], self.ssl_senario,
                                                       self.num_labels_per_class, self.num_of_client)

            if self.ssl_senario != 'server':
                ssl_s_split_type = 'iid'
                if self.s_split_type is not None:
                    ssl_s_split_type = self.s_split_type
                # ssl_s_split_type='dir'
                print('ssl_s_split_type', ssl_s_split_type)
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

            if self.local_test is None:
                _, u_train_data = data_simulation(self.u_train['x'],
                                                  self.u_train['y'].tolist(),
                                                  self.num_of_client,
                                                  self.split_type,
                                                  self.weights,
                                                  self.alpha,
                                                  self.min_size,
                                                  self.class_per_client)
            else:
                _, u_train_data, test_local_data = data_simulation_localtest(self.data_folder, self.u_train['x'],
                                                                             self.u_train['y'].tolist(),
                                                                             self.test_data['x'],
                                                                             self.test_data['y'],
                                                                             self.num_of_client,
                                                                             self.split_type,
                                                                             self.weights,
                                                                             self.alpha,
                                                                             self.min_size,
                                                                             self.class_per_client, test=True)




        else:
            _, train_data = data_simulation(self.train_data['x'],
                                            self.train_data['y'],
                                            self.num_of_client,
                                            self.split_type,
                                            self.weights,
                                            self.alpha,
                                            self.min_size,
                                            self.class_per_client)
        logger.info("Complete CIFAR100 data simulation")
        if self.is_ssl:
            save_dict(s_train_data, s_train_data_path)
            save_dict(u_train_data, u_train_data_path)
        else:
            save_dict(train_data, train_data_path)

        if not os.path.exists(test_data_path):
            save_dict(test_data, test_data_path)
        if self.local_test and not os.path.exists(local_test_data_path):
            save_dict(test_local_data, local_test_data_path)

    def convert_data_to_json(self):
        pass
