import argparse
import collections
import copy
import logging
import math
import numpy as np
import os
import pdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from easyfl.client import BaseClient
from easyfl.datasets.utils.util import save_dict, load_dict
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb
from easyfl.protocol import codec
from easyfl.tracking import metric
from easyfl.tracking.client import init_tracking
from easyfl.tracking.evaluation import model_size
from groups.models.ema import ModelEMA
from groups.models.model import WNet

logger = logging.getLogger(__name__)

CPU = "cpu"
NUM_CLASS = 10

class BaseClientSSL(BaseClient):
    def __init__(self,
                 cid,
                 conf,
                 s_train_data,
                 u_train_data,
                 test_data,
                 device,
                 data_path,
                 sleep_time=0,
                 is_remote=False,
                 local_port=23000,
                 server_addr="localhost:22999",
                 tracker_addr="localhost:12666"):

        super(BaseClientSSL, self).__init__(cid, conf, None, test_data, device, sleep_time,
                                            is_remote, local_port, server_addr, tracker_addr)
        self.s_train_data = s_train_data
        self.u_train_data = u_train_data
        self.s_train_loader = None
        self.u_train_loader = None
        self.class_dis_dir = None
        self.global_model = None
        if data_path is not None:
            self.class_dis_dir = os.path.join(data_path, 'class_distribution')
        self.get_bs(conf)

    # get unsupervised and supervised data batchsize
    def get_bs(self, conf):
        self.s_bs = conf.batch_size
        self.num_step = round(self.s_train_data.size(self.cid) / self.s_bs)
        if self.num_step<=1:
            self.num_step=2
        self.u_bs = self.u_train_data.size(self.cid) // self.num_step


    def pretrain_setup1(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        self.model.train()
        device_count = torch.cuda.device_count()
        
        self.model=self.model.to(device)
        if self.global_model is None:
            self.global_model = copy.deepcopy(self.model)
        else:
            self.global_model.load_state_dict(self.model.state_dict())
        
        self.global_model = self.global_model.to(self.device)
        for param in self.global_model.parameters():
            param.requires_grad = False

        loss_fn = self.load_loss_fn(conf)
        optimizer = self.load_optimizer(conf)
        if self.u_train_loader is None:
            self.u_train_loader = self.u_load_loader(conf)
        if self.s_train_loader is None:
            self.s_train_loader = self.s_load_loader(conf)

        return loss_fn, optimizer

    def train(self, conf, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()

        if conf.is_distributed:
            labeled_epoch = 0
            unlabeled_epoch = 0

        self.loss_fn, self.t_optimizer = self.pretrain_setup1(conf, device)
       
        labeled_iter = iter(self.s_train_loader)
        unlabeled_iter = iter(self.u_train_loader)
        unsup_criterion = nn.KLDivLoss(reduction='none')

        wnet = WNet(conf.num_classes, 100, 1).to(device)
        wnet.train()
        optimizer_wnet = torch.optim.Adam(wnet.parameters(), lr=0.001)

        self.train_loss = []

        for step in range(0, conf.total_steps):
            try:
                batched_s_x, batched_s_y = labeled_iter.__next__()
            except:
                if conf.is_distributed:
                    labeled_epoch += 1
               
                labeled_iter = iter(self.s_train_loader)
                
                batched_s_x, batched_s_y = next(labeled_iter)

            try:
                (batched_w_x, batched_stron_x), batch_u_y = unlabeled_iter.__next__()
            except:
                if conf.is_distributed:
                    unlabeled_epoch += 1
                #    self.u_train_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(self.u_train_loader)
                #(batched_w_x, batched_stron_x), batch_u_y = unlabeled_iter.__next__()
                (batched_w_x, batched_stron_x), batch_u_y = next(unlabeled_iter)

                # uda teacher
            bs = len(batched_s_x)
            weak_x, strong_x, super_x, super_y, batch_u_y= batched_w_x.to(device), batched_stron_x.to(device), batched_s_x.to(
                device), batched_s_y.to(device),batch_u_y.to(device) ,
            # u_y = batch_u_y.to(device)
            inputs = torch.cat((super_x, weak_x, strong_x), dim=0)
            
            t_out = self.model(inputs)

            t_logits_s = t_out[:bs]
            t_logits_uw, t_logits_us = t_out[bs:].chunk(2, dim=0)
            del t_out

            # Teacher Label loss
            t_loss_l = self.loss_fn(t_logits_s.contiguous(), super_y)
            with torch.no_grad():
                w_weight=wnet(t_logits_uw.softmax(1))
                norm=torch.sum(w_weight)

            coef = 1.0 * math.exp(-5 * (1 - min(step / 10, 1)) ** 2)

            ##unsupervised loss
            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / 1.0, dim=-1)
            max_probs, targets_u = torch.max(soft_pseudo_label, dim=-1)
            un_loss = (F.cross_entropy(t_logits_us.softmax(1), targets_u,
                                       reduction='none'))
            un_loss.reshape(un_loss.shape[0], 1)

            mask = max_probs.ge(0.6).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )

            if norm!=0:
                t_loss_uda=t_loss_l+coef*(torch.sum(un_loss*w_weight)/norm)
            else:
                t_loss_uda = t_loss_l + coef * (torch.sum(un_loss * w_weight))

            weight_u = 8 * min(1., (step + 1) / conf.uda_steps)
            #t_loss = t_loss_l + weight_u * t_loss_u
            t_loss = t_loss_l + t_loss_uda

            #t_loss = t_loss_l

            self.t_optimizer.zero_grad()
            t_loss.backward()
            self.t_optimizer.step()
            # self.t_scheduler.step()

            self.model.zero_grad()
            self.train_loss.append(float(t_loss.item()))
            # print('step:',step)
            if step % 100 == 0:
                logger.info("Client {}, local steps: {},t_loss:{}"
                            .format(self.cid, step, t_loss.item()))

        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        self.model.train()
        self.model.to(device)
        loss_fn = self.load_loss_fn(conf)
        optimizer = self.load_optimizer(conf)
        if self.u_train_loader is None:
            self.u_train_loader = self.u_load_loader(conf)
        if self.s_train_loader is None:
            self.s_train_loader = self.s_load_loader(conf)

        return loss_fn, optimizer

    def s_load_loader(self, conf):
        """Load the training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        s_train_loader = self.s_train_data.loader(self.s_bs, self.cid, shuffle=True, seed=conf.seed)
        # u_train_loader=self.u_train_data.loader(conf.batch_size, self.cid, shuffle=True, seed=conf.seed)
        return s_train_loader

    def u_load_loader(self, conf):
        """Load the training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        u_train_loader = self.u_train_data.loader(self.u_bs, self.cid, shuffle=True, seed=conf.seed)
        # u_train_loader=self.u_train_data.loader(conf.batch_size, self.cid, shuffle=True, seed=conf.seed)
        return u_train_loader

    def construct_upload_request(self):
        """Construct client upload request for training updates and testing results.

        Returns:
            :obj:`UploadRequest`: The upload request defined in protobuf to unify local and remote operations.
        """
        data = codec.marshal(server_pb.Performance(accuracy=self.test_accuracy, loss=self.test_loss))
        typ = common_pb.DATA_TYPE_PERFORMANCE
        try:
            if self._is_train:
                data = codec.marshal(copy.deepcopy(self.compressed_model))
                typ = common_pb.DATA_TYPE_PARAMS
                data_size = self.u_train_data.size(self.cid) + self.s_train_data.size(self.cid)
            else:
                data_size = 1 if not self.test_data else self.test_data.size(self.cid)
        except KeyError:
            # When the datasize cannot be get from dataset, default to use equal aggregate
            data_size = 1

        m = self._tracker.get_client_metric().to_proto() if self._tracker else common_pb.ClientMetric()
        return server_pb.UploadRequest(
            task_id=self.conf.task_id,
            round_id=self.conf.round_id,
            client_id=self.cid,
            content=server_pb.UploadContent(
                data=data,
                type=typ,
                data_size=data_size,
                metric=m,
            ),
        )

    def operate(self, model, conf, index, is_train=True):
        """A wrapper over operations (training/testing) on clients.

        Args:
            model (nn.Module): Model for operations.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            index (int): Client index in the client list, for retrieving data. TODO: improvement.
            is_train (bool): The flag to indicate whether the operation is training, otherwise testing.
        """
        try:
            # Load the data index depending on server request
            self.cid = self.u_train_data.users[index]
        except IndexError:
            logger.error("Data index exceed the available data, abort training")
            return

        if self.conf.track and self._tracker is None:
            self._tracker = init_tracking(init_store=False)

        if is_train:
            logger.info("Train on data index {}, client: {}".format(index, self.cid))
            self.run_train(model, conf)
        else:
            logger.info("Test on data index {}, client: {}".format(index, self.cid))
            self.run_test(model, conf)


class FedDureClientSSL(BaseClientSSL):
    def __init__(self,
                 cid,
                 conf,
                 s_train_data,
                 u_train_data,
                 test_data,
                 device,
                 data_path=None,
                 sleep_time=0,
                 is_remote=False,
                 local_port=23000,
                 server_addr="localhost:22999",
                 tracker_addr="localhost:12666"):

        super(FedDureClientSSL, self).__init__(cid, conf, s_train_data, u_train_data, test_data, device,
                                                     data_path,
                                                     sleep_time, is_remote, local_port, server_addr, tracker_addr)
        self.s_train_data = s_train_data
        self.u_train_data = u_train_data
        self.s_train_loader = None
        self.u_train_loader = None
        self.teacher_model = None
        self.accuracy = 0
        self.get_bs(conf)
        self.class_dis_dir = None

        if data_path is not None:
            self.class_dis_dir = os.path.join(data_path, 'class_distribution')
        
    def get_bs(self, conf):
        self.s_bs = conf.batch_size
        self.num_step = round(self.s_train_data.size(self.cid) / self.s_bs)


        if self.num_step<=2:
            self.num_step=4
        self.u_bs = self.u_train_data.size(self.cid) // self.num_step
        if self.u_bs==0:
            self.u_bs = self.u_train_data.size(self.cid)
        if self.u_bs>=400:
            self.u_bs = self.u_bs // 4

    def get_ema_model(self, model):
        self.teacher_EMA = ModelEMA(model, 0.8, self.device)
        self.teacher_model = self.teacher_EMA.ema

    def sync_teacher_student(self, teacher_model):
        if self.compressed_model:
            self.teacher_model.load_state_dict(teacher_model.state_dict())
            self.model.load_state_dict(teacher_model.state_dict())
        else:
            
            self.teacher_model = copy.deepcopy(teacher_model)
            self.model = copy.deepcopy(teacher_model)


    def run_pretrain(self, teacher_model, conf):
        self.conf = conf
        if conf.track:
            self._tracker.set_client_context(conf.task_id, conf.round_id, self.cid)

        self._is_train = True

        self.sync_teacher_student(teacher_model)
        self.track(metric.TRAIN_DOWNLOAD_SIZE, model_size(teacher_model))

        self.pre_train()
       
        self.loss_fn, self.s_optimizer, self.t_optimizer, self.s_scheduler, self.t_scheduler = self.pretrain_setup(conf,
                                                                                                                   self.device)
        
    def run_emsemble_train(self, teacher_model, conf, is_teacher_warm=True):
        
        if is_teacher_warm:
            self.run_pretrain(teacher_model, conf)
            self.train_teacher(conf, self.device)
            self.post_train()

            self.track(metric.TRAIN_ACCURACY, self.train_accuracy)
            self.track(metric.TRAIN_LOSS, self.train_loss)
            self.track(metric.TRAIN_TIME, self.train_time)

            # self.compression()

            self.track(metric.TRAIN_UPLOAD_SIZE, model_size(self.teacher_model))

            # self.encryption()

            return self.upload_teacher(self.teacher_model)
        else:
            # assert self.model is not None
            self.run_pretrain(teacher_model, conf)
            self.train_label_teacher(conf,self.device)
            self.train_step(conf, self.device)
            

            self.encryption()

            return self.upload()

    def test_local(self):
        # self.accuracy=self._local_test(self.teacher_model,self.device)

        self.accuracy = self.weighted_local_test(self.teacher_model, self.device)

    def _local_test(self, model, device):
        model.eval()
        model.to(device)
        test_local_loss = 0
        correct = 0
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batched_x, batched_y in self.test_data.loader(self.conf.test_batch_size, self.cid, shuffle=False,
                                                              seed=self.conf.seed):
                x = batched_x.to(device)
                y = batched_y.to(device)
                log_probs = model(x)

                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                test_local_loss += loss.item()
            test_data_size = self.test_data.size(self.cid)
            test_local_loss /= test_data_size
            accuracy = 100.00 * correct / test_data_size
        return accuracy

    def weighted_local_test(self, model, device):
        model.eval()
        model.to(device)
        test_local_loss = 0
        correct = 0
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        gt_size_per_class = np.zeros((10, 2))
        with torch.no_grad():
            for batched_x, batched_y in self.test_data.loader(self.conf.test_batch_size, shuffle=False,
                                                              seed=self.conf.seed):
                x = batched_x.to(device)
                y = batched_y.to(device)
                log_probs = model(x)

                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                test_local_loss += loss.item()

                batched_y = batched_y.long().cpu()
                y_pred = y_pred.long().cpu()
                for i in range(len(batched_y)):
                    label = batched_y[i]
                    gt_size_per_class[int(label)][0] += 1
                    pred = y_pred[i]
                    if pred == label:
                        gt_size_per_class[int(label)][1] += 1

            test_data_size = self.test_data.size()
            test_local_loss /= test_data_size
            # accuracy = 100.00 * correct / test_data_size

        if self.class_dis_dir is not None:
            class_dis_json = load_dict(self.class_dis_dir)
            class_dis = class_dis_json[self.cid]
            norm_class_dis = np.array(class_dis) / sum(class_dis)
            accuracy_per_class = gt_size_per_class[:, 1] / gt_size_per_class[:, 0]

            accuracy = (accuracy_per_class * norm_class_dis).sum()

        else:
            accuracy = 100.00 * correct / test_data_size
        logger.info('Client {},Personalized Model testing -- Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.cid, test_local_loss, correct, test_data_size, float(accuracy) * 100))

        return accuracy

    def test(self, conf, device=CPU):
        """Execute client testing.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        begin_test_time = time.time()
        self.model.eval()
        self.model.to(device)
        loss_fn = self.load_loss_fn(conf)

        if self.test_loader is None:
            self.test_loader = self.test_data.loader(conf.test_batch_size, shuffle=False, seed=conf.seed)
        # TODO: make evaluation metrics a separate package and apply it here.
        self.test_loss = 0
        correct = 0

        gt_size_per_class = np.zeros((10, 2))
        with torch.no_grad():
            for batched_x, batched_y in self.test_loader:
                x = batched_x.to(device)
                y = batched_y.to(device)
                log_probs = self.model(x)
                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                self.test_loss += loss.item()

                batched_y = batched_y.long().cpu()
                y_pred = y_pred.long().cpu()
                for i in range(len(batched_y)):
                    label = batched_y[i]
                    gt_size_per_class[int(label)][0] += 1
                    pred = y_pred[i]
                    if pred == label:
                        gt_size_per_class[int(label)][1] += 1

            test_size = self.test_data.size()
            self.test_loss /= test_size

            class_dis_json = load_dict(self.class_dis_dir)
            class_dis = class_dis_json[self.cid]
            norm_class_dis = np.array(class_dis) / sum(class_dis)
            accuracy_per_class = gt_size_per_class[:, 1] / gt_size_per_class[:, 0]

            self.test_accuracy = 100 * float((accuracy_per_class * norm_class_dis).sum())

            # self.test_accuracy = 100.0 * float(correct) / test_size

        logger.info('Client {},Global Model testing -- Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.cid, self.test_loss, correct, test_size, self.test_accuracy))

        self.test_time = time.time() - begin_test_time
        self.model = self.model.cpu()

    def download_teacher(self, teacher_model):
        if self.compressed_model:
            self.teacher_model.load_state_dict(teacher_model.state_dict())
        else:
            self.teacher_model = copy.deepcopy(teacher_model)

    def upload_teacher(self, model):
        data = codec.marshal(server_pb.Performance(accuracy=self.test_accuracy, loss=self.test_loss))
        typ = common_pb.DATA_TYPE_PERFORMANCE
        try:
            if self._is_train:

                data = codec.marshal(copy.deepcopy(self.teacher_model))

                typ = common_pb.DATA_TYPE_PARAMS
                data_size = self.u_train_data.size(self.cid) + self.s_train_data.size(self.cid)
            else:
                data_size = 1 if not self.test_data else self.test_data.size(self.cid)
        except KeyError:
            # When the datasize cannot be get from dataset, default to use equal aggregate
            data_size = 1

        m = self._tracker.get_client_metric().to_proto() if self._tracker else common_pb.ClientMetric()
        return server_pb.UploadRequest(
            task_id=self.conf.task_id,
            round_id=self.conf.round_id,
            client_id=self.cid,
            content=server_pb.UploadContent(
                data=data,
                type=typ,
                data_size=data_size,
                metric=m,
            ),

        )

    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()

        self.teacher_model.train()
        self.teacher_model.to(device)
        self.model.train()
        self.model.to(device)

        loss_fn = self.load_loss_fn(conf)

        s_optimizer = self.load_optimizer(conf)
        t_optimizer = self.load_teacher_optimizer(conf)
        #t_optimizer = self.load_optimizer(conf)

        t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                      conf.scheduler.warmup_steps,
                                                      conf.total_steps)
        s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                      conf.scheduler.warmup_steps,
                                                      conf.total_steps,
                                                      conf.scheduler.student_wait_steps
                                                      )

        if self.u_train_loader is None:
            self.u_train_loader = self.u_load_loader(conf)
        if self.s_train_loader is None:
            self.s_train_loader = self.s_load_loader(conf)

        return loss_fn, s_optimizer, t_optimizer, s_scheduler, t_scheduler

    def s_load_loader(self, conf):
        """Load the training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        s_train_loader = self.s_train_data.loader(self.s_bs, self.cid, shuffle=True, seed=conf.seed)
        # u_train_loader=self.u_train_data.loader(conf.batch_size, self.cid, shuffle=True, seed=conf.seed)
        self.s_num_per_classes=self.s_train_data.get_num_per_class(self.cid)

        return s_train_loader

    def u_load_loader(self, conf):
        """Load the training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        u_train_loader = self.u_train_data.loader(self.u_bs, self.cid, shuffle=True, seed=conf.seed)
        self.u_num_per_classes = self.u_train_data.get_num_per_class(self.cid)
        # u_train_loader=self.u_train_data.loader(conf.batch_size, self.cid, shuffle=True, seed=conf.seed)
        return u_train_loader

    def construct_upload_request(self):
        """ Construct client upload request for training updates and testing results.

        Returns:
            :obj:`UploadRequest`: The upload request defined in protobuf to unify local and remote operations.
        """
        
        dot_num=sum(self.dot_product)/float(5)
        data = codec.marshal(server_pb.Performance(accuracy=self.test_accuracy, loss=self.test_loss))
        typ = common_pb.DATA_TYPE_PERFORMANCE
        try:
            if self._is_train:
                # data = codec.marshal(copy.deepcopy(self.compressed_model))
                
                data = codec.marshal([copy.deepcopy(self.teacher_model), self.accuracy, dot_num])
                # ema_data = codec.marshal(copy.deepcopy(self.emsemble_EMA.ema))
                typ = common_pb.DATA_TYPE_PARAMS
                data_size = self.u_train_data.size(self.cid) + self.s_train_data.size(self.cid)
            else:
                data_size = 1 if not self.test_data else self.test_data.size(self.cid)
        except KeyError:
            # When the datasize cannot be get from dataset, default to use equal aggregate
            data_size = 1

        #del self.weights
        m = self._tracker.get_client_metric().to_proto() if self._tracker else common_pb.ClientMetric()
        return server_pb.UploadRequest(
            task_id=self.conf.task_id,
            round_id=self.conf.round_id,
            client_id=self.cid,
            content=server_pb.UploadContent(
                data=data,
                type=typ,
                data_size=data_size,
                metric=m,
            ),
        )

    def operate(self, model, conf, index, is_train=True):
        """A wrapper over operations (training/testing) on clients.

        Args:
            model (nn.Module): Model for operations.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            index (int): Client index in the client list, for retrieving data. TODO: improvement.
            is_train (bool): The flag to indicate whether the operation is training, otherwise testing.
        """
        try:
            # Load the data index depending on server request
            self.cid = self.u_train_data.users[index]
        except IndexError:
            logger.error("Data index exceed the available data, abort training")
            return

        if self.conf.track and self._tracker is None:
            self._tracker = init_tracking(init_store=False)

        if is_train:
            logger.info("Train on data index {}, client: {}".format(index, self.cid))
            self.run_train(model, conf)
        else:
            logger.info("Test on data index {}, client: {}".format(index, self.cid))
            self.run_test(model, conf)


    def load_teacher_optimizer(self, conf):
        """Load training optimizer. Implemented Adam and SGD."""
        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.teacher_model.parameters(), lr=conf.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(self.teacher_model.parameters(),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        return optimizer

    def train_step(self, conf, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()

        if conf.is_distributed:
            labeled_epoch = 0
            unlabeled_epoch = 0

        labeled_iter = iter(self.s_train_loader)
        unlabeled_iter = iter(self.u_train_loader)

        self.train_loss = []
        self.dot_product = []
        for step in range(0, conf.total_steps):
            try:
                batched_s_x, batched_s_y = labeled_iter.__next__()
            except:
                if conf.is_distributed:
                    labeled_epoch += 1
                #    self.s_train_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(self.s_train_loader)
                batched_s_x, batched_s_y = labeled_iter.__next__()

            try:
                (batched_w_x, batched_stron_x), batch_u_y = unlabeled_iter.__next__()
            except:
                if conf.is_distributed:
                    unlabeled_epoch += 1
                #    self.u_train_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(self.u_train_loader)
                (batched_w_x, batched_stron_x), batch_u_y = unlabeled_iter.__next__()

            # uda teacher
            bs = len(batched_s_x)
            weak_x, strong_x, super_x, super_y = batched_w_x.to(device), batched_stron_x.to(device), batched_s_x.to(
                device), batched_s_y.to(device),
            # u_y = batch_u_y.to(device)
            inputs = torch.cat((super_x, weak_x, strong_x), dim=0)
            self.teacher_model = self.teacher_model.to(device)

            t_out = self.teacher_model(inputs)

            t_logits_s = t_out[:bs]
            t_logits_uw, t_logits_us = t_out[bs:].chunk(2, dim=0)
            del t_out

            # Teacher Label loss
            t_loss_l = self.loss_fn(t_logits_s.contiguous(), super_y)

            ##unsupervised loss
            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / 1.0, dim=-1)
            max_probs, targets_u = torch.max(soft_pseudo_label, dim=-1)

            mask = max_probs.ge(conf.teacher_thre).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
           
            # metric
            # psudo_acc = sum((u_y == targets_u).float()) / len(targets_u)
            weight_u = 8 * min(1., (step + 1) / conf.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            # ------Student Model First CALL-------
            s_logits_l = self.model(super_x)
            s_logits_us = self.model(weak_x)
            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), super_y)
            s_loss = self.loss_fn(s_logits_us, targets_u)

            self.s_optimizer.zero_grad()
            s_loss.backward()
            self.s_optimizer.step()
            self.s_scheduler.step()
            # print('student',self.s_scheduler.step(), self.s_optimizer.step())

            # Student Model Second Call
            with torch.no_grad():
                s_logits_l = self.model(super_x)

            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), super_y)

            dot_product = s_loss_l_new - s_loss_l_old
            # print('dot', dot_product)

            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            # print('t_loss_mpl',t_loss_mpl)
            t_loss = t_loss_uda + t_loss_mpl
            #t_loss = t_loss_uda
            self.dot_product.append(float(dot_product.item()))

            self.t_optimizer.zero_grad()
            t_loss.backward()
            self.t_optimizer.step()
            self.t_scheduler.step()

            self.model.zero_grad()
            self.teacher_model.zero_grad()

            self.train_loss.append(float(s_loss.item()) + float(t_loss.item()))
            if step % 500 == 0:
                logger.debug("Client {}, local steps: {}, s_loss: {},t_loss:{}"
                            .format(self.cid, step, s_loss.item(), t_loss.item()))

        self.train_time = time.time() - start_time
        logger.info("Client {},Train {}, Train Time: {}".format(self.cid, conf.total_steps, self.train_time))

    def train_teacher(self, conf, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()

        if conf.is_distributed:
            labeled_epoch = 0
            unlabeled_epoch = 0

        labeled_iter = iter(self.s_train_loader)
        unlabeled_iter = iter(self.u_train_loader)

        self.train_loss = []

        for step in range(0, conf.teacher_steps):
            try:
                batched_s_x, batched_s_y = labeled_iter.__next__()
            except:
                if conf.is_distributed:
                    labeled_epoch += 1
                #    self.s_train_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(self.s_train_loader)
                batched_s_x, batched_s_y = labeled_iter.__next__()

            try:
                (batched_w_x, batched_stron_x), batch_u_y = unlabeled_iter.__next__()
            except:
                if conf.is_distributed:
                    unlabeled_epoch += 1
                #    self.u_train_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(self.u_train_loader)
                (batched_w_x, batched_stron_x), batch_u_y = unlabeled_iter.__next__()

            # uda teacher
            bs = len(batched_s_x)
            weak_x, strong_x, super_x, super_y = batched_w_x.to(device), batched_stron_x.to(device), batched_s_x.to(
                device), batched_s_y.to(device),
            # u_y = batch_u_y.to(device)
            inputs = torch.cat((super_x, weak_x, strong_x), dim=0)

            t_out = self.teacher_model(inputs)

            t_logits_s = t_out[:bs]
            t_logits_uw, t_logits_us = t_out[bs:].chunk(2, dim=0)
            del t_out

            # Teacher Label loss
            t_loss_l = self.loss_fn(t_logits_s.contiguous(), super_y)

            ##unsupervised loss
            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / 1.0, dim=-1)
            max_probs, targets_u = torch.max(soft_pseudo_label, dim=-1)

            mask = max_probs.ge(conf.teacher_thre).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
            
            weight_u = 8 * min(1., (step + 1) / conf.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            t_loss = t_loss_uda

            self.t_optimizer.zero_grad()
            t_loss.backward()
            self.t_optimizer.step()
            self.t_scheduler.step()

            self.teacher_model.zero_grad()

            self.train_loss.append(float(t_loss.item()))
            if step % 100 == 0:
                logger.debug("Client {}, local steps: {},t_loss:{}"
                             .format(self.cid, step, t_loss.item()))

        self.train_time = time.time() - start_time
        logger.info(
            "Client {}, Warmup Train Steps {}, Train Time: {}".format(self.cid, conf.teacher_steps, self.train_time))

    def train_label_teacher(self, conf, device=CPU):
        """Execute client training.
        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()

        if conf.is_distributed:
            labeled_epoch = 0
            unlabeled_epoch = 0

        labeled_iter = iter(self.s_train_loader)
        unlabeled_iter = iter(self.u_train_loader)

        self.train_loss = []

        for step in range(0, conf.fine_tune_steps):
            try:
                batched_s_x, batched_s_y = labeled_iter.__next__()
            except:
                if conf.is_distributed:
                    labeled_epoch += 1
                #    self.s_train_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(self.s_train_loader)
                batched_s_x, batched_s_y = next(labeled_iter)

            try:
                (batched_w_x, batched_stron_x), batch_u_y = unlabeled_iter.__next__()
            except:
                if conf.is_distributed:
                    unlabeled_epoch += 1
                #    self.u_train_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(self.u_train_loader)
                (batched_w_x, batched_stron_x), batch_u_y = next(unlabeled_iter)

            # uda teacher
            bs = len(batched_s_x)
            weak_x, strong_x, super_x, super_y = batched_w_x.to(device), batched_stron_x.to(device), batched_s_x.to(
                device), batched_s_y.to(device),
            # u_y = batch_u_y.to(device)
            inputs = torch.cat((super_x, weak_x, strong_x), dim=0)

            t_out = self.teacher_model(inputs)

            t_logits_s = t_out[:bs]

            del t_out

            # Teacher Label loss

            t_loss = self.loss_fn(t_logits_s.contiguous(), super_y)

            self.t_optimizer.zero_grad()
            t_loss.backward()
            self.t_optimizer.step()
            self.t_scheduler.step()

            self.teacher_model.zero_grad()

            self.train_loss.append(float(t_loss.item()))
            if step % 100 == 0:
                logger.debug("Labeled Teacher fine tuning, Client {}, local steps: {},t_loss:{}"
                             .format(self.cid, step, t_loss.item()))

        self.train_time = time.time() - start_time
        logger.info(
            "Labeled Teacher fine tuning, Client {}, Warmup Train Steps {}, Train Time: {}".format(self.cid,
                                                                                                   conf.fine_tune_steps,
                                                                                                   self.train_time))


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        # print('current step',current_step,num_wait_steps)

        if current_step < num_wait_steps:
            # print('***current step**', current_step, num_wait_steps)
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)