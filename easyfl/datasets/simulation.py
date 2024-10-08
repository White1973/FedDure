import heapq
import logging
import math
import os
import pdb

import numpy as np
from easyfl.datasets.utils.util import save_dict
SIMULATE_IID = "iid"
SIMULATE_NIID_DIR = "dir"
SIMULATE_NIID_CLASS = "class"
SIMULATE_NIID_DIR_SSL="dir_ssl"
logger = logging.getLogger(__name__)


def shuffle(data_x, data_y):
    num_of_data = len(data_y)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    index = [i for i in range(num_of_data)]
    np.random.shuffle(index)
    data_x = data_x[index]
    data_y = data_y[index]
    return data_x, data_y


def equal_division(num_groups, data_x, data_y=None):
    """Partition data into multiple clients with equal quantity.

    Args:
        num_groups (int): THe number of groups to partition to.
        data_x (list[Object]): A list of elements to be divided.
        data_y (list[Object], optional): A list of data labels to be divided together with the data.

    Returns:
        list[list]: A list where each element is a list of data of a group/client.
        list[list]: A list where each element is a list of data label of a group/client.

    Example:
        >>> equal_division(3, list[range(9)])
        >>> ([[0,4,2],[3,1,7],[6,5,8]], [])
    """
    if data_y is not None:
        assert (len(data_x) == len(data_y))
        data_x, data_y = shuffle(data_x, data_y)
    else:
        np.random.shuffle(data_x)
    num_of_data = len(data_x)

    assert num_of_data > 0
    data_per_client = num_of_data // num_groups
    large_group_num = num_of_data - num_groups * data_per_client
    small_group_num = num_groups - large_group_num
    splitted_data_x = []
    splitted_data_y = []
    for i in range(small_group_num):
        base_index = data_per_client * i
        splitted_data_x.append(data_x[base_index: base_index + data_per_client])
        if data_y is not None:
            splitted_data_y.append(data_y[base_index: base_index + data_per_client])
    small_size = data_per_client * small_group_num
    data_per_client += 1
    for i in range(large_group_num):
        base_index = small_size + data_per_client * i
        splitted_data_x.append(data_x[base_index: base_index + data_per_client])
        if data_y is not None:
            splitted_data_y.append(data_y[base_index: base_index + data_per_client])

    return splitted_data_x, splitted_data_y


def quantity_hetero(weights, data_x, data_y=None):
    """Partition data into multiple clients with different quantities.
    The number of groups is the same as the number of elements of `weights`.
    The quantity of each group depends on the values of `weights`.

    Args:
        weights (list[float]): The targeted distribution of data quantities.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
        data_x (list[Object]): A list of elements to be divided.
        data_y (list[Object], optional): A list of data labels to be divided together with the data.

    Returns:
        list[list]: A list where each element is a list of data of a group/client.
        list[list]: A list where each element is a list of data label of a group/client.
        
    Example:
        >>> quantity_hetero([0.1, 0.2, 0.7], list(range(0, 10)))
        >>> ([[4], [8, 9], [6, 0, 1, 7, 3, 2, 5]], [])
    """
    # This is due to the float number in python,
    # e.g.sum([0.1,0.2,0.4,0.2,0.1]) is not exactly 1, but 1.0000000000000002.
    assert (round(sum(weights), 3) == 1)

    if data_y is not None:
        assert (len(data_x) == len(data_y))
        data_x, data_y = shuffle(data_x, data_y)
    else:
        np.random.shuffle(data_x)
    data_size = len(data_x)

    i = 0

    splitted_data_x = []
    splitted_data_y = []
    for w in weights:
        size = math.floor(data_size * w)
        splitted_data_x.append(data_x[i:i + size])
        if data_y is not None:
            splitted_data_y.append(data_y[i:i + size])
        i += size

    parts = len(weights)
    if i < data_size:
        remain = data_size - i
        for i in range(-remain, 0, 1):
            splitted_data_x[(-i) % parts].append(data_x[i])
            if data_y is not None:
                splitted_data_y[(-i) % parts].append(data_y[i])

    return splitted_data_x, splitted_data_y


def iid(data_x, data_y, num_of_clients, x_dtype, y_dtype):
    """Partition dataset into multiple clients with equal data quantity (difference is less than 1) randomly.

    Args:
        data_x (list[Object]): A list of data.
        data_y (list[Object]): A list of dataset labels.
        num_of_clients (int): The number of clients to partition to.
        x_dtype (numpy.dtype): The type of data.
        y_dtype (numpy.dtype): The type of data label.

    Returns:
        list[str]: A list of client ids.
        dict: The partitioned data, key is client id, value is the client data. e.g., {'client_1': {'x': [data_x], 'y': [data_y]}}.
    """
    data_x, data_y = shuffle(data_x, data_y)
    x_divided_list, y_divided_list = equal_division(num_of_clients, data_x, data_y)
    clients = []
    federated_data = {}
    client_class_dis = {}
    for i in range(num_of_clients):
        client_id = "f%07.0f" % (i)
        temp_client = {}
        temp_client['x'] = np.array(x_divided_list[i]).astype(x_dtype)
        temp_client['y'] = np.array(y_divided_list[i]).astype(y_dtype)
        federated_data[client_id] = temp_client
        clients.append(client_id)
        client_class_dis[client_id] = np.array([sum(temp_client['y'] == c) / len(temp_client['y']) for c in range(10)])
        #print(i, '--iid clients class num:', [sum(temp_client['y'] == c) for c in range(10)])
    return clients, federated_data

def iid_unified(data_x, data_y,test_x,test_y, num_of_clients, x_dtype, y_dtype):
    """Partition dataset into multiple clients with equal data quantity (difference is less than 1) randomly.

    Args:
        data_x (list[Object]): A list of data.
        data_y (list[Object]): A list of dataset labels.
        num_of_clients (int): The number of clients to partition to.
        x_dtype (numpy.dtype): The type of data.
        y_dtype (numpy.dtype): The type of data label.

    Returns:
        list[str]: A list of client ids.
        dict: The partitioned data, key is client id, value is the client data. e.g., {'client_1': {'x': [data_x], 'y': [data_y]}}.
    """
    data_x, data_y = shuffle(data_x, data_y)
    test_x,test_y=shuffle(test_x,test_y)
    num_class = np.amax(data_y) + 1

    x_divided_list, y_divided_list = equal_division(num_of_clients, data_x, data_y)
    x_test_list, y_test_list = equal_division(num_of_clients, test_x, test_y)
    clients = []
    federated_data = {}
    test_federated_data = {}
    client_class_dis = {}
    for i in range(num_of_clients):
        client_id = "f%07.0f" % (i)
        temp_client = {}
        temp_client['x'] = np.array(x_divided_list[i]).astype(x_dtype)
        temp_client['y'] = np.array(y_divided_list[i]).astype(y_dtype)
        test_temp_client={}
        test_temp_client['x'] = np.array(x_test_list[i]).astype(x_dtype)
        test_temp_client['y'] = np.array(y_test_list[i]).astype(y_dtype)
        federated_data[client_id] = temp_client
        clients.append(client_id)

        client_class_dis[client_id]=np.array([sum(temp_client['y']==c)/len(temp_client['y']) for c in range(10)])
        print(i,'--clients class num:',[sum(temp_client['y']==c) for c in range(10)] )
    return clients, federated_data, test_federated_data, client_class_dis


def non_iid_dirichlet(data_x, data_y, num_of_clients, alpha, min_size, x_dtype, y_dtype):
    """Partition dataset into multiple clients following the Dirichlet process.

    Args:
        data_x (list[Object]): A list of data.
        data_y (list[Object]): A list of dataset labels.
        num_of_clients (int): The number of clients to partition to.
        alpha (float): The parameter for Dirichlet process simulation.
        min_size (int): The minimum number of data size of a client.
        x_dtype (numpy.dtype): The type of data.
        y_dtype (numpy.dtype): The type of data label.

    Returns:
        list[str]: A list of client ids.
        dict: The partitioned data, key is client id, value is the client data. e.g., {'client_1': {'x': [data_x], 'y': [data_y]}}.
    """
    n_train = data_x.shape[0]
    current_min_size = 0
    num_class = np.amax(data_y) + 1
    data_size = data_y.shape[0]
    net_dataidx_map = {}


    while current_min_size < min_size:
        idx_batch = [[] for _ in range(num_of_clients)]
        for k in range(num_class):
            idx_k = np.where(data_y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_of_clients))
            # using the proportions from dirichlet, only selet those clients having data amount less than average
            proportions = np.array(
                [p * (len(idx_j) < data_size / num_of_clients) for p, idx_j in zip(proportions, idx_batch)])
            # scale proportions
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            current_min_size = min([len(idx_j) for idx_j in idx_batch])

    federated_data = {}
    clients = []
    for j in range(num_of_clients):
        np.random.shuffle(idx_batch[j])
        client_id = "f%07.0f" % j
        clients.append(client_id)
        temp = {}
        temp['x'] = np.array(data_x[idx_batch[j]]).astype(x_dtype)
        temp['y'] = np.array(data_y[idx_batch[j]]).astype(y_dtype)
        federated_data[client_id] = temp
        net_dataidx_map[client_id] = idx_batch[j]
    print_data_distribution(data_y, net_dataidx_map)
    return clients, federated_data





def non_iid_class(data_x, data_y, class_per_client, num_of_clients, x_dtype, y_dtype, stack_x=True):
    """Partition dataset into multiple clients based on label classes.
    Each client contains [1, n] classes, where n is the number of classes of a dataset.

    Note: Each class is divided into `ceil(class_per_client * num_of_clients / num_class)` parts
        and each client chooses `class_per_client` parts from each class to construct its dataset.

    Args:
        data_x (list[Object]): A list of data.
        data_y (list[Object]): A list of dataset labels.
        class_per_client (int): The number of classes in each client.
        num_of_clients (int): The number of clients to partition to.
        x_dtype (numpy.dtype): The type of data.
        y_dtype (numpy.dtype): The type of data label.
        stack_x (bool, optional): A flag to indicate whether using np.vstack or append to construct dataset.

    Returns:
        list[str]: A list of client ids.
        dict: The partitioned data, key is client id, value is the client data. e.g., {'client_1': {'x': [data_x], 'y': [data_y]}}.
    """
    num_class = np.amax(data_y) + 1
    all_index = []
    clients = []
    data_index_map = {}
    for i in range(num_class):
        # get indexes for all data with current label i at index i in all_index
        all_index.append(np.where(data_y == i)[0].tolist())

    federated_data = {}

    # total no. of parts
    total_amount = class_per_client * num_of_clients
    # no. of parts each class should be diveded into
    parts_per_class = math.ceil(total_amount / num_class)

    for i in range(num_of_clients):
        client_id = "f%07.0f" % (i)
        clients.append(client_id)
        data_index_map[client_id] = []
        data = {}
        data['x'] = np.array([])
        data['y'] = np.array([])
        federated_data[client_id] = data

    class_map = {}
    parts_consumed = []
    for i in range(num_class):
        class_map[i], _ = equal_division(parts_per_class, all_index[i])
        heapq.heappush(parts_consumed, (0, i))
    for i in clients:
        for j in range(class_per_client):
            class_chosen = heapq.heappop(parts_consumed)
            part_indexes = class_map[class_chosen[1]].pop(0)
            if len(federated_data[i]['x']) != 0:
                if stack_x:
                    federated_data[i]['x'] = np.vstack((federated_data[i]['x'], data_x[part_indexes])).astype(x_dtype)
                else:
                    federated_data[i]['x'] = np.append(federated_data[i]['x'], data_x[part_indexes]).astype(x_dtype)
                federated_data[i]['y'] = np.append(federated_data[i]['y'], data_y[part_indexes]).astype(y_dtype)
            else:
                federated_data[i]['x'] = data_x[part_indexes].astype(x_dtype)
                federated_data[i]['y'] = data_y[part_indexes].astype(y_dtype)
            heapq.heappush(parts_consumed, (class_chosen[0] + 1, class_chosen[1]))
            data_index_map[i].extend(part_indexes)
    print_data_distribution(data_y, data_index_map)
    return clients, federated_data

def non_iid_ssl_type(data_x,data_y,num_of_clients,x_dtype, y_dtype):
    ten_types_of_class_imbalanced_dist = [
        [0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15],  # type 0
        [0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03],  # type 1
        [0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03],  # type 2
        [0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03],  # type 3
        [0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02],  # type 4
        [0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03],  # type 5
        [0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03],  # type 6
        [0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03],  # type 7
        [0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15],  # type 8
        [0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50],  # type 9
    ]
    num_u=len(list(data_y))
    labels=list(set(data_y))
    num_u_per_client = int(num_u/num_of_clients)
    offset_per_label = {label: 0 for label in labels}
    federated_data={}
    clients=[]

    for cid in range(num_of_clients):
        x_unlabeled=[]
        y_unlabeled=[]
        client_id = "f%07.0f" % (cid)
        dist_type=cid%len(labels)
        freqs = np.random.choice(labels, num_u_per_client, p=ten_types_of_class_imbalanced_dist[dist_type])
        frq=[]
        temp_client={}

        for label in labels:
            idx_k = np.where(data_y == label)[0]
            data=data_x[idx_k]
            data_labels=data_y[idx_k]
            num_instances=len(freqs[freqs==label])
            frq.append(num_instances)
            if num_instances!=0:
                start=offset_per_label[label]
                end=offset_per_label[label]+num_instances
                x_unlabeled=[*x_unlabeled,*data[start:end]]
                y_unlabeled=[*y_unlabeled,*data_labels[start:end]]
                offset_per_label[label]=end
        temp_client['x']=np.array(x_unlabeled).astype(x_dtype)
        temp_client['y']=np.array(y_unlabeled).astype(y_dtype)

        federated_data[client_id]=temp_client
        clients.append(client_id)
    return clients,federated_data


def non_iid_ssl_type_partition_train_test(data_x,data_y,test_x,test_y,num_of_clients,x_dtype, y_dtype):
    ten_types_of_class_imbalanced_dist = [
        [0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15],  # type 0
        [0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03],  # type 1
        [0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03],  # type 2
        [0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03],  # type 3
        [0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02],  # type 4
        [0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03],  # type 5
        [0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03],  # type 6
        [0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03],  # type 7
        [0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15],  # type 8
        [0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50],  # type 9
    ]
    num_u=len(list(data_y))
    labels=list(set(data_y))

    test_num_u=len(list(test_x))
    test_labels=list(set(test_y))
    num_test_per_client=int(test_num_u/num_of_clients)

    num_u_per_client = int(num_u/num_of_clients)
    offset_per_label = {label: 0 for label in labels}
    test_offset_per_label={label:0 for label in test_labels}

    federated_data={}
    test_federated_data={}

    clients=[]
    client_class_dis = {}
    for cid in range(num_of_clients):
        x_unlabeled=[]
        y_unlabeled=[]

        test_x_u=[]
        test_y_u=[]

        client_id = "f%07.0f" % (cid)
        dist_type=cid%len(labels)

        freqs = np.random.choice(labels, num_u_per_client, p=ten_types_of_class_imbalanced_dist[dist_type])
        test_freqs=np.random.choice(labels,num_test_per_client,p=ten_types_of_class_imbalanced_dist[dist_type])

        client_class_dis[client_id]=ten_types_of_class_imbalanced_dist[dist_type]

        frq=[]
        temp_client={}

        test_frq=[]
        test_temp_client={}



        for label in labels:
            idx_k = np.where(data_y == label)[0]
            data=data_x[idx_k]
            data_labels=data_y[idx_k]
            num_instances=len(freqs[freqs==label])
            frq.append(num_instances)
            start=offset_per_label[label]
            end=offset_per_label[label]+num_instances
            x_unlabeled=[*x_unlabeled,*data[start:end]]
            y_unlabeled=[*y_unlabeled,*data_labels[start:end]]
            offset_per_label[label]=end

            test_idx_k=np.where(test_y==label)[0]
            test_data=test_x[test_idx_k]
            test_data_label=test_y[test_idx_k]
            test_num_instances=len(test_freqs[test_freqs==label])
            test_frq.append(test_num_instances)
            t_start=test_offset_per_label[label]
            t_end=test_offset_per_label[label]+test_num_instances
            test_x_u=[*test_x_u,*test_data[t_start:t_end]]
            test_y_u=[*test_y_u,*test_data_label[t_start:t_end]]
            test_offset_per_label[label]=end

        temp_client['x']=np.array(x_unlabeled).astype(x_dtype)
        temp_client['y']=np.array(y_unlabeled).astype(y_dtype)

        test_temp_client['x']=np.array(test_x_u).astype(x_dtype)
        test_temp_client['y']=np.array(test_y_u).astype(y_dtype)


        federated_data[client_id]=temp_client
        test_federated_data[client_id]=test_temp_client
        clients.append(client_id)
    return clients,federated_data,test_federated_data,client_class_dis


def non_iid_dirichlet_partition_train_test(data_x, data_y,test_x,test_y,num_of_clients, alpha, min_size, x_dtype, y_dtype):
    """Partition dataset into multiple clients following the Dirichlet process.

    Args:
        data_x (list[Object]): A list of data.
        data_y (list[Object]): A list of dataset labels.
        num_of_clients (int): The number of clients to partition to.
        alpha (float): The parameter for Dirichlet process simulation.
        min_size (int): The minimum number of data size of a client.
        x_dtype (numpy.dtype): The type of data.
        y_dtype (numpy.dtype): The type of data label.

    Returns:
        list[str]: A list of client ids.
        dict: The partitioned data, key is client id, value is the client data. e.g., {'client_1': {'x': [data_x], 'y': [data_y]}}.
    """
    n_train = data_x.shape[0]
    current_min_size = 0
    num_class = np.amax(data_y) + 1
    data_size = data_y.shape[0]
    net_dataidx_map = {}
    test_net_dataidx_map={}


    n_test=test_x.shape[0]
    test_size=data_y.shape[0]

    ClassDistClients=[]

    while current_min_size < min_size:
        idx_batch = [[] for _ in range(num_of_clients)]
        test_idx_batch=[[] for _ in range(num_of_clients)]
        for k in range(num_class):
            idx_k = np.where(data_y == k)[0]
            test_idx_k=np.where(test_y==k)[0]
            np.random.shuffle(test_idx_k)
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_of_clients))
            # using the proportions from dirichlet, only selet those clients having data amount less than average
            proportions = np.array(
                [p * (len(idx_j) < data_size / num_of_clients) for p, idx_j in zip(proportions, idx_batch)])
            # scale proportions
            _proportions = proportions / proportions.sum()
            ClassDistClients.append(_proportions)

            proportions = (np.cumsum(_proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            current_min_size = min([len(idx_j) for idx_j in idx_batch])

            #test_samples;
            test_proportions=(np.cumsum(_proportions)*len(test_idx_k)).astype(int)[:-1]
            test_idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(test_idx_batch, np.split(test_idx_k, test_proportions))]

    federated_data = {}
    test_federarted_data={}
    clients = []
    client_class_dis={}
    ClassDistClients=np.array(ClassDistClients).T
    for j in range(num_of_clients):
        np.random.shuffle(idx_batch[j])
        client_id = "f%07.0f" % j
        client_class_dis[client_id]=ClassDistClients[j].tolist()
        clients.append(client_id)
        temp = {}
        temp['x'] = np.array(data_x[idx_batch[j]]).astype(x_dtype)
        temp['y'] = np.array(data_y[idx_batch[j]]).astype(y_dtype)
        federated_data[client_id] = temp
        net_dataidx_map[client_id] = idx_batch[j]


        np.random.shuffle(test_idx_batch)
        test_temp={}
        test_temp['x']=np.array(test_x[test_idx_batch[j]]).astype(x_dtype)
        test_temp['y']=np.array(test_y[test_idx_batch[j]]).astype(y_dtype)
        test_federarted_data[client_id]=test_temp
        test_net_dataidx_map[client_id]=test_idx_batch[j]
    print_data_distribution(data_y, net_dataidx_map)
    print_data_distribution(test_y,test_net_dataidx_map)

    return clients, federated_data,test_federarted_data,client_class_dis

def data_simulation(data_x, data_y, num_of_clients, data_distribution, weights=None, alpha=0.5, min_size=10,
                    class_per_client=1, stack_x=True):
    """Simulate federated learning datasets by partitioning a data into multiple clients using different strategies.

    Args:
        data_x (list[Object]): A list of data.
        data_y (list[Object]): A list of dataset labels.
        num_of_clients (int): The number of clients to partition to.
        data_distribution (str): The ways to partition the dataset, options:
            `iid`: Partition dataset into multiple clients with equal quantity (difference is less than 1) randomly.
            `dir`: partition dataset into multiple clients following the Dirichlet process.
            `class`: partition dataset into multiple clients based on classes.
        weights: list, for simulating data quantity heterogeneity
            If None, each client are simulated with same data quantity
            Note: num_of_clients should be divisible by len(weights)
        weights (list[float], optional): The targeted distribution of data quantities.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            When `weights=None`, the data quantity of clients only depends on data_distribution.
        alpha (float, optional): The parameter for Dirichlet process simulation.
            It is only applicable when data_distribution is `dir`.
        min_size (int, optional): The minimum number of data size of a client.
            It is only applicable when data_distribution is `dir`.
        class_per_client (int): The number of classes in each client.
            It is only applicable when data_distribution is `class`.
        stack_x (bool, optional): A flag to indicate whether using np.vstack or append to construct dataset.
            It is only applicable when data_distribution is `class`.

    Raise:
        ValueError: When the simulation method `data_distribution` is not supported.

    Returns:
        list[str]: A list of client ids.
        dict: The partitioned data, key is client id, value is the client data. e.g., {'client_1': {'x': [data_x], 'y': [data_y]}}.
    """
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    x_dtype = data_x.dtype
    y_dtype = data_y.dtype

    if weights is not None:
        assert num_of_clients % len(weights) == 0
        num_of_clients = num_of_clients // len(weights)

    if data_distribution == SIMULATE_IID:
        group_client_list, group_federated_data = iid(data_x, data_y, num_of_clients, x_dtype, y_dtype)
    elif data_distribution == SIMULATE_NIID_DIR:
        group_client_list, group_federated_data = non_iid_dirichlet(data_x, data_y, num_of_clients, alpha, min_size,
                                                                    x_dtype, y_dtype)
    elif data_distribution == SIMULATE_NIID_CLASS:
        group_client_list, group_federated_data = non_iid_class(data_x, data_y, class_per_client, num_of_clients,
                                                                x_dtype,
                                                                y_dtype, stack_x=stack_x)
    elif data_distribution==SIMULATE_NIID_DIR_SSL:
        group_client_list, group_federated_data=non_iid_ssl_type(data_x,data_y,num_of_clients,x_dtype,y_dtype)
    else:
        raise ValueError("Simulation type not supported")
    if weights is None:
        return group_client_list, group_federated_data

    clients = []
    federated_data = {}
    cur_key = 0

    for i in group_client_list:
        current_client = group_federated_data[i]
        input_lists, label_lists = quantity_hetero(weights, current_client['x'], current_client['y'])
        for j in range(len(input_lists)):
            client_id = "f%07.0f" % (cur_key)
            temp_client = {}
            temp_client['x'] = np.array(input_lists[j]).astype(x_dtype)
            temp_client['y'] = np.array(label_lists[j]).astype(y_dtype)
            federated_data[client_id] = temp_client
            clients.append(client_id)
            cur_key += 1
    return clients, federated_data

def data_simulation_localtest(data_folder,data_x, data_y,test_x,test_y, num_of_clients, data_distribution, weights=None, alpha=0.5, min_size=10,
                    class_per_client=1, stack_x=True,test=True):
    """Simulate federated learning datasets by partitioning a data into multiple clients using different strategies.

    Args:
        data_x (list[Object]): A list of data.
        data_y (list[Object]): A list of dataset labels.
        num_of_clients (int): The number of clients to partition to.
        data_distribution (str): The ways to partition the dataset, options:
            `iid`: Partition dataset into multiple clients with equal quantity (difference is less than 1) randomly.
            `dir`: partition dataset into multiple clients following the Dirichlet process.
            `class`: partition dataset into multiple clients based on classes.
        weights: list, for simulating data quantity heterogeneity
            If None, each client are simulated with same data quantity
            Note: num_of_clients should be divisible by len(weights)
        weights (list[float], optional): The targeted distribution of data quantities.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            When `weights=None`, the data quantity of clients only depends on data_distribution.
        alpha (float, optional): The parameter for Dirichlet process simulation.
            It is only applicable when data_distribution is `dir`.
        min_size (int, optional): The minimum number of data size of a client.
            It is only applicable when data_distribution is `dir`.
        class_per_client (int): The number of classes in each client.
            It is only applicable when data_distribution is `class`.
        stack_x (bool, optional): A flag to indicate whether using np.vstack or append to construct dataset.
            It is only applicable when data_distribution is `class`.

    Raise:
        ValueError: When the simulation method `data_distribution` is not supported.

    Returns:
        list[str]: A list of client ids.
        dict: The partitioned data, key is client id, value is the client data. e.g., {'client_1': {'x': [data_x], 'y': [data_y]}}.
    """
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    test_x=np.array(test_x)
    test_y=np.array(test_y)
    x_dtype = data_x.dtype
    y_dtype = data_y.dtype
    assert test==True
    if weights is not None:
        assert num_of_clients % len(weights) == 0
        num_of_clients = num_of_clients // len(weights)

    client_class_dis=None
    if data_distribution == SIMULATE_IID:
        group_client_list, group_federated_data,test_federarted_data,client_class_dis= iid_unified(data_x, data_y,test_x,test_y, num_of_clients, x_dtype, y_dtype)
        #group_client_list, group_federated_data = iid(data_x, data_y, num_of_clients, x_dtype, y_dtype)
        test_federarted_data=None
    elif data_distribution == SIMULATE_NIID_DIR:
        group_client_list, group_federated_data,test_federarted_data,client_class_dis = non_iid_dirichlet_partition_train_test(data_x, data_y,test_x,test_y,num_of_clients, alpha, min_size,
                                                                    x_dtype, y_dtype)
    elif data_distribution==SIMULATE_NIID_DIR_SSL:
        group_client_list, group_federated_data, test_federarted_data, client_class_dis=non_iid_ssl_type_partition_train_test(data_x, data_y,test_x,test_y,num_of_clients,x_dtype, y_dtype)
    else:
        raise ValueError("Simulation type not supported")
    if client_class_dis:
        class_dis_path=os.path.join(data_folder,'class_distribution')
        save_dict(client_class_dis,class_dis_path)

    if weights is None:
        return group_client_list, group_federated_data,test_federarted_data

    clients = []
    federated_data = {}
    test_data={}
    cur_key = 0

    for i in group_client_list:
        current_client = group_federated_data[i]
        test_current_client=test_federarted_data[i]
        input_lists, label_lists = quantity_hetero(weights, current_client['x'], current_client['y'])
        test_input_lists,test_label_lists=quantity_hetero(weights,test_current_client['x'],test_current_client['y'])
        for j in range(len(input_lists)):
            client_id = "f%07.0f" % (cur_key)
            temp_client = {}
            temp_client['x'] = np.array(input_lists[j]).astype(x_dtype)
            temp_client['y'] = np.array(label_lists[j]).astype(y_dtype)
            federated_data[client_id] = temp_client

            test_temp_client={}
            test_temp_client['x']=np.array(test_input_lists[j]).astype(x_dtype)
            test_temp_client['y']=np.array(test_label_lists[j]).astype(y_dtype)
            test_data[client_id]=test_temp_client
            clients.append(client_id)
            cur_key += 1
    return clients, federated_data,test_data


def print_data_distribution(data_y, data_index_map):
    """Log the distribution of client datasets."""
    data_distribution = {}
    for index, dataidx in data_index_map.items():
        unique_values, counts = np.unique(data_y[dataidx], return_counts=True)
        distribution = {unique_values[i]: counts[i] for i in range(len(unique_values))}
        data_distribution[index] = distribution
    logger.info(data_distribution)
    return data_distribution

#semi-supervised federated learning data split
def split_s_and_u(data_x, data_y, ssl_senario, num_labels_per_class, num_of_clients):
    s_by_label, u_by_label = {}, {}
    if ssl_senario == 'server':
        num_s = num_labels_per_class
    if ssl_senario == 'client_part':
        num_s = num_labels_per_class * num_of_clients
    labels = np.unique(data_y)
    data_y = np.array(data_y)

    data_by_label = {}
    for label in labels:
        idx = np.where(data_y[:]==label)[0] #返回行索引
        data_by_label[label] = {
            'x': data_x[idx],
            'y': data_y[idx]
        }

    s_by_label, u_by_label = {}, {}

    for label, data in data_by_label.items():
        if len(s_by_label)==0 or len(u_by_label)==0:
            s_by_label = {
                'x': data['x'][:num_s],
                'y': data['y'][:num_s]
            }
            u_by_label = {
                'x': data['x'][num_s:],
                'y': data['y'][num_s:]
            }
        else:
            s_by_label = {
                'x': np.vstack((s_by_label['x'], data['x'][:num_s])),
                'y': np.hstack((s_by_label['y'], data['y'][:num_s]))
            }
            u_by_label = {
                'x': np.vstack((u_by_label['x'], data['x'][num_s:])),
                'y': np.hstack((u_by_label['y'], data['y'][num_s:]))
            }
    if s_by_label['x'].shape[0]==10:
        s_by_label['x']=s_by_label['x'].flatten()
        u_by_label['x']=u_by_label['x'].flatten()

    num_u = len(u_by_label['y'])
    return s_by_label, u_by_label


def split_s_and_u_dir(train_data, ssl_senario, num_labels_per_class, num_of_clients):
    #num_s = num_labels_per_class
    s_train_data, u_train_data = {},{}
    num_label, num_unlabel =0, 0

    for cid in train_data.keys():

        labels = np.unique(train_data[cid]['y'])
        data_y = np.array(train_data[cid]['y'])
        data_x = np.array(train_data[cid]['x'])
        #print(cid,":",labels)

        data_by_label = {}
        for label in labels:
            idx = np.where(data_y[:] == label)[0]  # 返回行索引
            data_by_label[label] = {
                'x': data_x[idx],
                'y': data_y[idx]
            }

        s_by_label, u_by_label = {}, {}
        u_num, s_num=[], []
        for label, data in data_by_label.items():
            num_s = num_labels_per_class if data['y'].shape[0] > num_labels_per_class else data['y'].shape[0]
            s_num.append(num_s)
            if data['x'].shape[0]>num_s:
                u_num.append(data['x'].shape[0]-num_s)
            else:
                u_num.append(0)
            if len(s_by_label) == 0 or len(u_by_label) == 0:
                s_by_label = {
                    'x': data['x'][:num_s],
                    'y': data['y'][:num_s]
                }
                try:
                    u_by_label = {
                        'x': data['x'][num_s:],
                        'y': data['y'][num_s:]
                    }
                except:
                    pass

            else:
                try:
                    s_by_label = {
                        'x': np.vstack((s_by_label['x'], data['x'][:num_s])),
                        'y': np.hstack((s_by_label['y'], data['y'][:num_s]))
                    }
                    u_by_label = {
                        'x': np.vstack((u_by_label['x'], data['x'][num_s:])),
                        'y': np.hstack((u_by_label['y'], data['y'][num_s:]))
                    }
                    if s_by_label['x'].shape[0] == 10:
                        s_by_label['x'] = s_by_label['x'].flatten()
                        u_by_label['x'] = u_by_label['x'].flatten()
                except:
                    s_by_label = {
                        'x': np.concatenate((s_by_label['x'].reshape(-1, 1), data['x'][:num_s].reshape(-1, 1)), axis=0),
                        'y': np.concatenate((s_by_label['y'].reshape(-1, 1), data['y'][:num_s].reshape(-1, 1)), axis=0)
                    }

                    u_by_label = {
                        'x': np.concatenate((u_by_label['x'].reshape(-1, 1), data['x'][num_s:].reshape(-1, 1)), axis=0),
                        'y': np.concatenate((u_by_label['y'].reshape(-1, 1), data['y'][num_s:].reshape(-1, 1)), axis=0)
                    }
                    s_by_label['x'] = s_by_label['x'].flatten()
                    u_by_label['x'] = u_by_label['x'].flatten()

                    s_by_label['y'] = s_by_label['y'].flatten()
                    u_by_label['y'] = u_by_label['y'].flatten()

        s_train_data[cid] = s_by_label
        u_train_data[cid] = u_by_label
        print(cid,"---labeled:", s_num, " unlabeled:", u_num)

        num_label += s_by_label['y'].shape[0]
        num_unlabel += u_by_label['y'].shape[0]
        print(cid,"---labeled:", s_by_label['y'].shape[0], " unlabeled:", u_by_label['y'].shape[0])
    #print('num', num_label, num_unlabel)

    #pdb.set_trace()
    return s_train_data, u_train_data
