import importlib
import json
import logging
import os
import pdb


from easyfl.datasets.dataset import FederatedTensorDataset
from easyfl.datasets.utils.base_dataset import BaseDataset, CIFAR10, CIFAR100, TINY_IMAGENET,CINIC10, COVID19
from easyfl.datasets.utils.util import load_dict

logger = logging.getLogger(__name__)


def read_dir(data_dir):
    clients = []
    groups = []
    data = {}

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(dataset_name, train_data_dir, test_data_dir):
    """Load datasets from data directories.

    Args:
        dataset_name (str): The name of the dataset.
        train_data_dir (str): The directory of training data.
        test_data_dir (str): The directory of testing data.

    Returns:
        list[str]: A list of client ids.
        list[str]: A list of group ids for dataset with hierarchies.
        dict: A dictionary of training data, e.g., {"id1": {"x": data, "y": label}, "id2": {"x": data, "y": label}}.
        dict: A dictionary of testing data. The format is same as training data for FEMNIST and Shakespeare datasets.
            For CIFAR datasets, the format is {"x": data, "y": label}, for centralized testing in the server.
    """
    if dataset_name == CIFAR10 or dataset_name == CIFAR100 or dataset_name==TINY_IMAGENET or dataset_name==CINIC10 or dataset_name==COVID19:
        train_data = load_dict(train_data_dir)
        test_data = load_dict(test_data_dir)
        return [], [], train_data, test_data

    # Data in the directories are `json` files with keys `users` and `user_data`.
    print(test_data_dir)

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def load_data(root,
              dataset_name,
              num_of_clients,
              split_type,
              min_size,
              class_per_client,
              data_amount,
              iid_fraction,
              user,
              train_test_split,
              quantity_weights,
              alpha,
              ssl_senario,
              num_labels_per_class,
              is_ssl,
              local_test=None,
              s_split_type=None):
    """Simulate and load federated datasets.

    Args:
        root (str): The root directory where datasets stored.
        dataset_name (str): The name of the dataset. It currently supports: femnist, shakespeare, cifar10, and cifar100.
            Among them, femnist and shakespeare are adopted from LEAF benchmark.
        num_of_clients (int): The targeted number of clients to construct.
        split_type (str): The type of statistical simulation, options: iid, dir, and class.
            `iid` means independent and identically distributed data.
            `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
            `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
            `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
        min_size (int): The minimal number of samples in each client.
            It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
        class_per_client (int): The number of classes in each client. Only applicable when the split_type is 'class'.
        data_amount (float): The fraction of data sampled for LEAF datasets.
            e.g., 10% means that only 10% of total dataset size are used.
        iid_fraction (float): The fraction of the number of clients used when the split_type is 'iid'.
        user (bool): A flag to indicate whether partition users of the dataset into train-test groups.
            Only applicable to LEAF datasets.
            True means partitioning users of the dataset into train-test groups.
            False means partitioning each users' samples into train-test groups.
        train_test_split (float): The fraction of data for training; the rest are for testing.
            e.g., 0.9 means 90% of data are used for training and 10% are used for testing.
        quantity_weights (list[float]): The targeted distribution of quantities to simulate data quantity heterogeneity.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            The `num_of_clients` should be divisible by `len(weights)`.
            None means clients are simulated with the same data quantity.
        alpha (float): The parameter for Dirichlet distribution simulation, applicable only when split_type is `dir`.

    Returns:
        dict: A dictionary of training data, e.g., {"id1": {"x": data, "y": label}, "id2": {"x": data, "y": label}}.
        dict: A dictionary of testing data.
        function: A function to preprocess training data.
        function: A function to preprocess testing data.
        torchvision.transforms.transforms.Compose: Training data transformation.
        torchvision.transforms.transforms.Compose: Testing data transformation.
    """
    user_str = "user" if user else "sample"
    #dataset, split_type, num_of_client, min_size, class_per_client, alpha, quantity_weights
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, num_labels_per_class, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights,s_split_type) #cifar10_dir_100_10_1_0.5_0
    #os.path.realpath(__file__) 文本绝对路径'/opt/disk/bsk/projects/fedlearn/Code/EasyFL/easyfl/datasets/data.py'

    dir_path = os.path.dirname(os.path.realpath(__file__)) #'/opt/disk/bsk/projects/fedlearn/Code/EasyFL/easyfl/datasets'
    dataset_file = os.path.join(dir_path, "data_process", "{}.py".format(dataset_name))
    if not os.path.exists(dataset_file):
        logger.error("Please specify a valid process file path for process_x and process_y functions.")
    dataset_path = "easyfl.datasets.data_process.{}".format(dataset_name)
    dataset_lib = importlib.import_module(dataset_path) #module:dataset_path
    process_x = getattr(dataset_lib, "process_x", None) #获取dataset_lib中"process_x"属性值，没有就返回None
    process_y = getattr(dataset_lib, "process_y", None) #只针对feminist and shakepeare type(torch.long)
    transform_train = getattr(dataset_lib, "transform_train", None)
    transform_test = getattr(dataset_lib, "transform_test", None)   #

    data_dir = os.path.join(root, dataset_name) #root:'/opt/disk/bsk/projects/fedlearn/Data'
    if not data_dir:
        os.makedirs(data_dir)

    if is_ssl:
        s_train_data_dir = os.path.join(data_dir, setting, "train_s_{}_{}".format(ssl_senario,num_labels_per_class))
        u_train_data_dir = os.path.join(data_dir, setting, "train_u_{}_{}".format(ssl_senario,num_labels_per_class))
    else:
        train_data_dir = os.path.join(data_dir, setting, "train")
    test_data_dir = os.path.join(data_dir, setting, "test")
    data_path = os.path.join(data_dir, setting)
    if local_test:
        local_test_data_dir=os.path.join(data_dir,setting,"local_test")
    if not os.path.exists(test_data_dir) or (is_ssl and not os.path.exists(u_train_data_dir)) or (is_ssl==False and not os.path.exists(train_data_dir)):
        dataset_class_path = "easyfl.datasets.{}.{}".format(dataset_name, dataset_name)
        dataset_class_lib = importlib.import_module(dataset_class_path)
        class_name = dataset_name.capitalize() #首字母大写
        dataset = getattr(dataset_class_lib, class_name)(root=data_dir, #'/opt/disk/bsk/projects/fedlearn/Code/EasyFL/easyfl/datasets/cidar10/Cifar10'
                                                         fraction=data_amount,
                                                         split_type=split_type,
                                                         user=user,
                                                         iid_user_fraction=iid_fraction,
                                                         train_test_split=train_test_split,
                                                         minsample=min_size,
                                                         num_of_client=num_of_clients,
                                                         class_per_client=class_per_client,
                                                         setting_folder=setting,
                                                         alpha=alpha,
                                                         weights=quantity_weights,
                                                         ssl_senario=ssl_senario,
                                                         num_labels_per_class=num_labels_per_class,
                                                         is_ssl=is_ssl,local_test=local_test,s_split_type=s_split_type) #
        try:
            filename = f"{setting}.zip"
            dataset.download_packaged_dataset_and_extract(filename)
            logger.info(f"Downloaded packaged dataset {dataset_name}: {filename}")
        except Exception as e:
            logger.info(f"Failed to download packaged dataset: {e.args}")

        # CIFAR10 generate data in setup() stage, LEAF related datasets generate data in sampling()
        if (is_ssl and not os.path.exists(u_train_data_dir)) or (not is_ssl and not os.path.exists(train_data_dir)):
            dataset.setup()
        if (is_ssl and not os.path.exists(u_train_data_dir)) or (not is_ssl and not os.path.exists(train_data_dir)):
            dataset.sampling()


    if is_ssl:

        users, train_groups, s_train_data, test_data = read_data(dataset_name, s_train_data_dir, test_data_dir)
        # import pdb
        # pdb.set_trace()
        if local_test:
            _, _, u_train_data, local_test_data = read_data(dataset_name, u_train_data_dir, local_test_data_dir)
        else:
            _, _, u_train_data, _ = read_data(dataset_name, u_train_data_dir, test_data_dir)
            local_test_data=None

        return s_train_data, u_train_data, test_data,local_test_data, process_x, process_y, transform_train, transform_test, data_path
    else:
        users, train_groups, train_data, test_data = read_data(dataset_name, train_data_dir, test_data_dir) #Cifar10:100,2
        return train_data, test_data, process_x, process_y, transform_train, transform_test

def construct_datasets(root,
                       dataset_name,
                       num_of_clients,
                       split_type,
                       min_size,
                       class_per_client,
                       data_amount,
                       iid_fraction,
                       user,
                       train_test_split,
                       quantity_weights,
                       alpha,
                       ssl_senario,
                       num_labels_per_class,
                       is_ssl,
                       local_test=None,
                       s_split_type=None):
    """Construct and load provided federated learning datasets.

    Args:
        root (str): The root directory where datasets stored.
        dataset_name (str): The name of the dataset. It currently supports: femnist, shakespeare, cifar10, and cifar100.
            Among them, femnist and shakespeare are adopted from LEAF benchmark.
        num_of_clients (int): The targeted number of clients to construct.
        split_type (str): The type of statistical simulation, options: iid, dir, and class.
            `iid` means independent and identically distributed data.
            `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
            `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
            `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
        min_size (int): The minimal number of samples in each client.
            It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
        class_per_client (int): The number of classes in each client. Only applicable when the split_type is 'class'.
        data_amount (float): The fraction of data sampled for LEAF datasets.
            e.g., 10% means that only 10% of total dataset size are used.
        iid_fraction (float): The fraction of the number of clients used when the split_type is 'iid'.
        user (bool): A flag to indicate whether partition users of the dataset into train-test groups.
            Only applicable to LEAF datasets.
            True means partitioning users of the dataset into train-test groups.
            False means partitioning each users' samples into train-test groups.
        train_test_split (float): The fraction of data for training; the rest are for testing.
            e.g., 0.9 means 90% of data are used for training and 10% are used for testing.
        quantity_weights (list[float]): The targeted distribution of quantities to simulate data quantity heterogeneity.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            The `num_of_clients` should be divisible by `len(weights)`.
            None means clients are simulated with the same data quantity.
        alpha (float): The parameter for Dirichlet distribution simulation, applicable only when split_type is `dir`.
        num_labels_per_class: Federated senario, the number labeled data per class instances

    Returns:
        :obj:`FederatedDataset`: Training dataset.
        :obj:`FederatedDataset`: Testing dataset.
    """
    if is_ssl:
        s_train_data, u_train_data, test_data,local_test_data, process_x, process_y, transform_train, transform_test, data_path = load_data(root,
                                                                                                                                            dataset_name,
                                                                                                                                            num_of_clients,
                                                                                                                                            split_type,
                                                                                                                                            min_size,
                                                                                                                                            class_per_client,
                                                                                                                                            data_amount,
                                                                                                                                            iid_fraction,
                                                                                                                                            user,
                                                                                                                                            train_test_split,
                                                                                                                                            quantity_weights,
                                                                                                                                            alpha,
                                                                                                                                            ssl_senario,
                                                                                                                                            num_labels_per_class,
                                                                                                                                            is_ssl,
                                                                                                                                            local_test,
                                                                                                                                            s_split_type)

    else:
        train_data, test_data, process_x, process_y, transform_train, transform_test = load_data(root,
                                                                                                 dataset_name,
                                                                                                 num_of_clients,
                                                                                                 split_type,
                                                                                                 min_size,
                                                                                                 class_per_client,
                                                                                                 data_amount,
                                                                                                 iid_fraction,
                                                                                                 user,
                                                                                                 train_test_split,
                                                                                                 quantity_weights,
                                                                                                 alpha,
                                                                                                 ssl_senario,
                                                                                                 num_labels_per_class,
                                                                                                 is_ssl,
                                                                                                 local_test)


    # CIFAR datasets are simulated.

    test_simulated = True
    if dataset_name == CIFAR10 or dataset_name == CIFAR100 or dataset_name==TINY_IMAGENET or dataset_name==CINIC10 or dataset_name==COVID19:
        test_simulated = False
    if dataset_name==CIFAR10 or dataset_name == CIFAR100 or dataset_name==TINY_IMAGENET or dataset_name==COVID19:
        from easyfl.datasets.data_process.cifar10 import TransformFixMatch
    elif dataset_name==CINIC10:
        from easyfl.datasets.data_process.cinic10 import TransformFixMatch
    if is_ssl:

        if ssl_senario == 'server':
            s_train_data = FederatedTensorDataset(s_train_data,
                                        simulated=False,
                                        do_simulate=False,
                                        process_x=process_x,
                                        process_y=process_y,
                                        transform=transform_train)
        else:
            s_train_data = FederatedTensorDataset(s_train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        process_x=process_x,
                                        process_y=process_y,
                                        transform=transform_train)
        
        u_train_data = FederatedTensorDataset(u_train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        process_x=process_x,
                                        process_y=process_y,
                                        transform=TransformFixMatch())
    else:
        train_data = FederatedTensorDataset(train_data,
                                            simulated=True,
                                            do_simulate=False,
                                            process_x=process_x,
                                            process_y=process_y,
                                            transform=transform_train)

    test_data = FederatedTensorDataset(test_data,
                                       simulated=test_simulated,
                                       do_simulate=False,
                                       process_x=process_x,
                                       process_y=process_y,
                                       transform=transform_test)
    if is_ssl and local_test_data is not None:
        local_test_data = FederatedTensorDataset(local_test_data,
                                           simulated=True,
                                           do_simulate=False,
                                           process_x=process_x,
                                           process_y=process_y,
                                           transform=transform_test)


    if is_ssl:

        return s_train_data, u_train_data, test_data,local_test_data, data_path
    else:
        return train_data, test_data