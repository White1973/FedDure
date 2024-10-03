import importlib
import logging
import pdb

import numpy as np
import os
import random
import sys
import time
import torch
from easydict import EasyDict
from omegaconf import OmegaConf
from os import path

from easyfl.client.base import BaseClient
from easyfl.coordinator import Coordinator
from easyfl.datasets import TEST_IN_SERVER
from easyfl.datasets.data import construct_datasets
from easyfl.distributed import dist_init, get_device
from easyfl.models.model import load_model
from easyfl.server.base import BaseServer
from easyfl.simulation.system_hetero import resource_hetero_simulation

logger = logging.getLogger(__name__)


class Coordinator_fssl(Coordinator):
    """Coordinator manages federated learning server and client.
    A single instance of coordinator is initialized for each federated learning task
    when the package is imported.
    """

    def __init__(self):
        self.registered_model = False
        self.registered_dataset = False
        self.registered_server = False
        self.registered_client = False
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.conf = None
        self.model = None
        self._model_class = None
        self.server = None
        self._server_class = None
        self.clients = None
        self._client_class = None
        self.tracker = None
        super(Coordinator_fssl, self).__init__()

    def init(self, conf, init_all=False):
        """Initialize coordinator

        Args:
            conf (omegaconf.dictconfig.DictConfig): Internal configurations for federated learning.
            init_all (bool): Whether initialize dataset, model, server, and client other than configuration.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        conf = init_conf(conf)


        init_logger(conf.tracking.log_level)

        _set_random_seed(conf.seed)

        self.init_conf(conf)

        _set_random_seed(conf.seed)

        self.server_client_registered()
        if init_all:
            self.init_dataset()

            self.init_model()

            self.init_server()
            if self.conf.data.is_ssl:
                self.init_clients_ssl()
            else:
                self.init_clients()

    def run(self):
        """Run the coordinator and the federated learning process.
        Initialize `torch.distributed` if distributed training is configured.
        """
        start_time = time.time()

        if self.conf.is_distributed:
            dist_init(
                self.conf.distributed.backend,
                self.conf.distributed.init_method,
                self.conf.distributed.world_size,
                self.conf.distributed.rank,
                self.conf.distributed.local_rank,
            )

        self.server.start(self.model, self.clients)
        self.print_("Total training time {:.1f}s".format(time.time() - start_time))

    def init_conf(self, conf):
        """Initialize coordinator configuration.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Configurations.
        """
        #
        self.conf = conf
        self.conf.is_distributed = (self.conf.gpu > 1)
        if self.conf.gpu == 0:
            self.conf.device = "cpu"
        elif self.conf.gpu == 1:
            self.conf.device = 0
        else:

            self.conf.device = get_device(self.conf.gpu, self.conf.distributed.world_size,
                                          self.conf.distributed.local_rank)
        self.print_("Configurations: {}".format(self.conf))

    def server_client_registered(self):
        client_type = self.conf.client_name
        if client_type is not None:
            client_class_path = "groups.client_ssl.{}".format('base_ssl')
            client_class_lib = importlib.import_module(client_class_path)
            _C = getattr(client_class_lib, client_type)
            # if client_type=='SSL':
            #    from groups.client_ssl.base_ssl import BaseClientSSL
            self.register_client(_C)

        server_type = self.conf.server_name
        if server_type is not None:
            server_class_path = "groups.server_ssl.{}".format('base_ssl')
            server_class_lib = importlib.import_module(server_class_path)
            _S = getattr(server_class_lib, server_type)
            self.register_server(_S)

    def init_dataset(self):
        """Initialize datasets. Use provided datasets if not registered."""
        if self.conf.data.is_ssl:
            self.s_train_data, self.u_train_data, self.test_data, self.local_test_data, self.data_path = construct_datasets(
                self.conf.data.root,
                self.conf.data.dataset,
                self.conf.data.num_of_clients,
                self.conf.data.split_type,
                self.conf.data.min_size,
                self.conf.data.class_per_client,
                self.conf.data.data_amount,
                self.conf.data.iid_fraction,
                self.conf.data.user,
                self.conf.data.train_test_split,
                self.conf.data.weights,
                self.conf.data.alpha,
                self.conf.data.ssl_senario,
                self.conf.data.num_labels_per_class,
                self.conf.data.is_ssl,
                local_test=self.conf.data.local_test,
                s_split_type=self.conf.data.s_split_type)

            # print(f"Total labeled training data amount: {self.s_train_data.total_size()}")
            # print(f"Total unlabeled training data amount: {self.u_train_data.total_size()}")

        else:
            self.train_data, self.test_data = construct_datasets(self.conf.data.root,
                                                                 self.conf.data.dataset,
                                                                 self.conf.data.num_of_clients,
                                                                 self.conf.data.split_type,
                                                                 self.conf.data.min_size,
                                                                 self.conf.data.class_per_client,
                                                                 self.conf.data.data_amount,
                                                                 self.conf.data.iid_fraction,
                                                                 self.conf.data.user,
                                                                 self.conf.data.train_test_split,
                                                                 self.conf.data.weights,
                                                                 self.conf.data.alpha,
                                                                 self.conf.data.ssl_senario,
                                                                 self.conf.data.num_labels_per_class,
                                                                 self.conf.data.is_ssl)
        #     print(f"Total training data amount: {self.train_data.total_size()}")
        #
        # print(f"Total testing data amount: {self.test_data.total_size()}")


    def init_model(self):
        """Initialize model instance."""
        if not self.registered_model:
            self._model_class = load_model(self.conf.model.type)

        # model_class is None means model is registered as instance, no need initialization
        if self._model_class:
            self.model = self._model_class(num_classes=self.conf.model.num_classes)

    def init_server(self):
        """Initialize a server instance."""
        if not self.registered_server:
            self._server_class = BaseServer

        kwargs = {
            "is_remote": self.conf.is_remote,
            "local_port": self.conf.local_port
        }

        if self.conf.test_mode == TEST_IN_SERVER:
            kwargs["test_data"] = self.test_data
            if self.val_data:
                kwargs["val_data"] = self.val_data
        elif self.conf.test_mode == "test_in_all" or self.conf.test_mode == "weighted_test_in_all":
            kwargs["test_data"] = self.test_data
            if self.val_data:
                kwargs["val_data"] = self.val_data

        self.server = self._server_class(self.conf, **kwargs)

    def init_clients(self):
        """Initialize client instances, each represent a federated learning client."""
        if not self.registered_client:
            self._client_class = BaseClient

        # Enforce system heterogeneity of clients.
        sleep_time = [0 for _ in self.train_data.users]
        if self.conf.resource_heterogeneous.simulate:
            sleep_time = resource_hetero_simulation(self.conf.resource_heterogeneous.fraction,
                                                    self.conf.resource_heterogeneous.hetero_type,
                                                    self.conf.resource_heterogeneous.sleep_group_num,
                                                    self.conf.resource_heterogeneous.level,
                                                    self.conf.resource_heterogeneous.total_time,
                                                    len(self.train_data.users))

        client_test_data = self.test_data
        if self.conf.test_mode == TEST_IN_SERVER:
            client_test_data = None

        self.clients = [self._client_class(u,
                                           self.conf.client,
                                           self.train_data,
                                           client_test_data,
                                           self.conf.device,
                                           **{"sleep_time": sleep_time[i]})
                        for i, u in enumerate(self.train_data.users)]

        self.print_("Clients in total: {}".format(len(self.clients)))

    def init_clients_ssl(self):
        """Initialize client instances, each represent a federated learning client."""
        assert self.conf.data.is_ssl == True
        if not self.registered_client:
            self._client_class = BaseClient

        # Enforce system heterogeneity of clients.
        sleep_time = [0 for _ in self.u_train_data.users]
        if self.conf.resource_heterogeneous.simulate:
            sleep_time = resource_hetero_simulation(self.conf.resource_heterogeneous.fraction,
                                                    self.conf.resource_heterogeneous.hetero_type,
                                                    self.conf.resource_heterogeneous.sleep_group_num,
                                                    self.conf.resource_heterogeneous.level,
                                                    self.conf.resource_heterogeneous.total_time,
                                                    len(self.u_train_data.users))

        client_test_data = self.test_data
        if self.conf.test_mode == TEST_IN_SERVER:
            client_test_data = None
        elif self.conf.test_mode == "test_in_all":
            client_test_data = self.local_test_data
        if self.conf.data.ssl_senario == 'client_part':
            self.clients = [self._client_class(u,
                                               self.conf.client,
                                               self.s_train_data,
                                               self.u_train_data,
                                               client_test_data,
                                               self.conf.device,
                                               data_path=self.data_path,
                                               **{"sleep_time": sleep_time[i]})
                            for i, u in enumerate(self.u_train_data.users)]
        else:
            pass

        self.print_("Clients in total: {}".format(len(self.clients)))

    def init_client(self):
        """Initialize client instance.

        Returns:
            :obj:`BaseClient`: The initialized client instance.
        """
        if not self.registered_client:
            self._client_class = BaseClient

        # Get a random client if not specified
        if self.conf.index:
            user = self.train_data.users[self.conf.index]
        else:
            user = random.choice(self.train_data.users)

        return self._client_class(user,
                                  self.conf.client,
                                  self.train_data,
                                  self.test_data,
                                  self.conf.device,
                                  is_remote=self.conf.is_remote,
                                  local_port=self.conf.local_port,
                                  server_addr=self.conf.server_addr,
                                  tracker_addr=self.conf.tracker_addr)

    def start_server(self, args):
        """Start a server service for remote training.

        Server controls the model and testing dataset if configured to test in server.

        Args:
            args (argparse.Namespace): Configurations passed in as arguments, it is merged with configurations.
        """
        if args:
            self.conf = OmegaConf.merge(self.conf, args.__dict__)

        if self.conf.test_mode == TEST_IN_SERVER:
            self.init_dataset()

        self.init_model()

        self.init_server()

        self.server.start_service()

    def start_client(self, args):
        """Start a client service for remote training.

        Client controls training datasets.

        Args:
            args (argparse.Namespace): Configurations passed in as arguments, it is merged with configurations.
        """

        if args:
            self.conf = OmegaConf.merge(self.conf, args.__dict__)

        self.init_dataset()

        client = self.init_client()

        client.start_service()

    def register_dataset(self, train_data, test_data, val_data=None):
        """Register datasets.

        Datasets should inherit from :obj:`FederatedDataset`, e.g., :obj:`FederatedTensorDataset`.

        Args:
            train_data (:obj:`FederatedDataset`): Training dataset.
            test_data (:obj:`FederatedDataset`): Testing dataset.
            val_data (:obj:`FederatedDataset`): Validation dataset.
        """
        self.registered_dataset = True
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def register_model(self, model):
        """Register customized model for federated learning.

        Args:
            model (nn.Module): PyTorch model, both class and instance are acceptable.
                Use model class when there is no specific arguments to initialize model.
        """
        self.registered_model = True
        if not isinstance(model, type):
            self.model = model
        else:
            self._model_class = model

    def register_server(self, server):
        """Register a customized federated learning server.

        Args:
            server (:obj:`BaseServer`): Customized federated learning server.
        """
        self.registered_server = True
        self._server_class = server

    def register_client(self, client):
        """Register a customized federated learning client.

        Args:
            client (:obj:`BaseClient`): Customized federated learning client.
        """
        self.registered_client = True
        self._client_class = client

    def print_(self, content):
        """Log the content only when the server is primary server.

        Args:
            content (str): The content to log.
        """
        if self._is_primary_server():
            logger.info(content)

    def _is_primary_server(self):
        """Check whether current running server is the primary server.

        In standalone or remote training, the server is primary.
        In distributed training, the server on `rank0` is primary.
        """
        return not self.conf.is_distributed or self.conf.distributed.rank == 0


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_conf(conf=None):
    """Initialize configuration for EasyFL. It overrides and supplements default configuration loaded from config.yaml
    with the provided configurations.

    Args:
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    # here = path.abspath(path.dirname(__file__))
    # config_file = path.join(here, '../easyfl/config.yaml')
    return EasyDict(conf)


def load_config(file, conf=None):
    """Load and merge configuration from file and input

    Args:
        file (str): filename of the configuration.
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    config = OmegaConf.load(file)
    if conf is not None:
        config = OmegaConf.merge(config, conf)
    return config


def init_logger(log_level):
    """Initialize internal logger of EasyFL.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG
    """
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    log_level = logging.INFO if not log_level else log_level
    root_logger.setLevel(log_level)

    file_path = os.path.join(os.getcwd(), "logs")
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)
    file_path = path.join(file_path, "train" + time.strftime(".%m_%d_%H_%M_%S") + ".log")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
