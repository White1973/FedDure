import argparse

import copy
import logging
import pdb
import random
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from easyfl.datasets import TEST_IN_SERVER
from easyfl.distributed.distributed import CPU
from easyfl.protocol import codec
from easyfl.server import BaseServer
from easyfl.server.base import MODEL, DATA_SIZE
from easyfl.tracking import metric
from easyfl.utils.float import rounding
import matplotlib.pyplot as plt
import pylab as pl
logger = logging.getLogger(__name__)

EMBED = "embed"
CONTROLS="controls"

class BaseServerSSL(BaseServer):
    def __init__(self,
                 conf,
                 test_data=None,
                 val_data=None,
                 is_remote=False,
                 local_port=22999):
        super(BaseServerSSL, self).__init__(conf, test_data, val_data, is_remote, local_port)

    def test(self):
        self.print_("--- start ALL testing ---")

        test_begin_time = time.time()
        test_results = {metric.TEST_ACCURACY: 0, metric.TEST_LOSS: 0, metric.TEST_TIME: 0}
        test_results_teacher = {metric.TEST_ACCURACY: 0, metric.TEST_LOSS: 0, metric.TEST_TIME: 0}
        if self.conf.test_mode == TEST_IN_SERVER:
            if self.is_primary_server():
                test_results_teacher = self.test_in_sever_TSModel(self._model, self.conf.device)
        elif self.conf.test_mode == "test_in_all" or self.conf.test_mode == "weighted_test_in_all":
            if self.is_primary_server():
                test_results_teacher = self.test_in_sever_TSModel(self._model, self.conf.device)

            ##TODO only sync teacher_model
            # self._model=self.model
            test_results = self.test_in_client()
            test_results[metric.TEST_TIME] = time.time() - test_begin_time
        else:
            test_results = self.test_in_client()
            test_results[metric.TEST_TIME] = time.time() - test_begin_time
        # test_results[metric.TEST_TIME] = time.time() - test_begin_time

        self.track_TStest_results(test_results_teacher, flag='teacher')
        #self.track_test_results(test_results)

    def test_in_sever_TSModel(self, model, device=CPU):
        model.eval()
        model.to(device)
        test_loss = 0
        correct = 0
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        features, labels = None, []
        with torch.no_grad():
            for batched_x, batched_y in self.test_data.loader(self.conf.server.batch_size, seed=self.conf.seed):
                x = batched_x.to(device)
                y = batched_y.to(device)
                x_features, log_probs = model(x, test=True)
                if (self._current_round + 1) == self.conf.server.rounds:
                    if features == None:
                        features = x_features
                    else:
                        features = torch.cat([features, x_features], dim=0)
                    # [features.append(x_features[index].cpu()) for index in range(y.shape[0])]
                    [labels.append(y[index].item()) for index in range(y.shape[0])]


                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                test_loss += loss.item()
            test_data_size = self.test_data.size()
            test_loss /= test_data_size
            accuracy = 100.00 * correct / test_data_size
            del features, labels


            test_results = {
                metric.TEST_ACCURACY: float(accuracy),
                metric.TEST_LOSS: float(test_loss)
            }
            return test_results

    def track_TStest_results(self, results, flag):
        self._cumulative_times.append(time.time() - self._start_time)
        self._accuracies.append(results[metric.TEST_ACCURACY])
        self.print_('TEACHER####Test loss: {:.2f}, Test accuracy: {:.2f}%'.format(
            results[metric.TEST_LOSS], results[metric.TEST_ACCURACY]))

    def aggregation(self):
        uploaded_content = self.get_client_uploads()

        list_models = list(uploaded_content[MODEL].values())
        models = [model for model in list_models]
        # teacher_model=[model['t_model'] for model in list_models]

        weights = list(uploaded_content[DATA_SIZE].values())


        model = self.aggregate(models, weights)
        # teacher_model=self.aggregate(teacher_model,weights)
        self.set_model(model, load_dict=True)
        # self.set_teacher_model(teacher_model,load_dict=True)



    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        uploaded_models = {}
        uploaded_weights = {}
        uploaded_metrics = []


        for client in self.grouped_clients:
            # Update client config before training
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self._current_round
            self.conf.client.is_distributed = self.conf.gpu > 0

            uploaded_request = client.run_train(self._compressed_model, self.conf.client)
            uploaded_content = uploaded_request.content

            model = self.decompression(codec.unmarshal(uploaded_content.data))
            uploaded_models[client.cid] = model

            uploaded_weights[client.cid] = uploaded_content.data_size
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

        self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)

class FedDureServerSSL(BaseServer):
    def __init__(self,
                 conf,
                 test_data=None,
                 val_data=None,
                 is_remote=False,
                 local_port=22999):
        super(FedDureServerSSL, self).__init__(conf, test_data, val_data, is_remote, local_port)

        self._losses =[]
        self.dotnum=[]
    def get_emsemble_model(self):
        self.teacher_model = copy.deepcopy(self._model)

    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        if self._current_round < self.conf.server.warm_rounds:
            uploaded_models = {}
            uploaded_weights = {}
            uploaded_metrics = []
            uploaded_dotnum = []
            uploaded_mweights= {}
            # self.get_emsemble_model()

            for client in self.grouped_clients:
                # Update client config before training
                self.conf.client.task_id = self.conf.task_id
                self.conf.client.round_id = self._current_round
                self.conf.client.is_distributed = self.conf.gpu > 0

                uploaded_request = client.run_emsemble_train(self.teacher_model,
                                                             self.conf.client, True)
                uploaded_content = uploaded_request.content

                #teacher_model = self.decompression(codec.unmarshal(uploaded_content.data))
                data = self.decompression(codec.unmarshal(uploaded_content.data))
                pdb.set_trace()
                teacher_model = data[0]
                #dotnum=data[2]

                uploaded_mweights[client.cid]=data[3]
                uploaded_dotnum.append(data[2])
                uploaded_models[client.cid] = teacher_model
                uploaded_weights[client.cid] = uploaded_content.data_size
                uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

            # logger.info("Memory_allocated:{}".format(torch.cuda.memory_allocated(0)/1024/1024))
            self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)
            if self._current_round%50==0 or self._current_round==self.conf.server.rounds-1:
                pdb.set_trace()
            self.print_(f'dotnum {sum(uploaded_dotnum)/float(len(uploaded_dotnum))}')
            self.aggregation()

        else:
            uploaded_models = {}
            uploaded_weights = {}
            uploaded_metrics = []
            accuracy_list = []
            uploaded_dotnum = []
            #uploaded_mweights = {}
            for client in self.grouped_clients:
                # Update client config before training
                self.conf.client.task_id = self.conf.task_id
                self.conf.client.round_id = self._current_round
                self.conf.client.is_distributed = self.conf.gpu > 0

                uploaded_request = client.run_emsemble_train(self.teacher_model,
                                                             self.conf.client, False)
                uploaded_content = uploaded_request.content

                teacher_model, accuracy, dot_num = self.decompression(codec.unmarshal(uploaded_content.data))
                #uploaded_mweights[client.cid] = m_weight
                uploaded_dotnum.append(dot_num)

                uploaded_models[client.cid] = teacher_model
                uploaded_weights[client.cid] = uploaded_content.data_size
                uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

                accuracy_list.append(accuracy)
            # logger.info("Memory_allocated:{}".format(torch.cuda.memory_allocated(0)/1024/1024))
            self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)
            self.dotnum.append(sum(uploaded_dotnum)/len(uploaded_dotnum))
            if self._current_round%50==0 and self._current_round>=100:
                pdb.set_trace()
            

    def start(self, model, clients):
        """Start federated learning process, including training and testing.

        Args:
            model (nn.Module): The model to train.
            clients (list[:obj:`BaseClient`]|list[str]): Available clients.
                Clients are actually client grpc addresses when in remote training.
        """
        # Setup
        self._start_time = time.time()
        self._reset()
        self.set_model(model)
        self.set_clients(clients)
        self.set_teacher_model(model)
        if self._should_track():
            self._tracker.create_task(self.conf.task_id, OmegaConf.to_container(self.conf))

        # Get initial testing accuracies
        if self.conf.server.test_all:
            if self._should_track():
                self._tracker.set_round(self._current_round)
            self.test()
            self.save_tracker()

        while not self.should_stop():
            self._round_time = time.time()

            self._current_round += 1
            self.print_("\n-------- round {} --------".format(self._current_round))

            # Train
            self.pre_train()
            self.train()
            self.post_train()

            # Test
            if self._do_every(self.conf.server.test_every, self._current_round, self.conf.server.rounds):
                self.pre_test()
                self.test()
                self.post_test()

            # Save Model
            self.save_model()

            self.track(metric.ROUND_TIME, time.time() - self._round_time)
            self.save_tracker()

        self.print_("Accuracies: {}".format(rounding(self._accuracies, 4)))
        self.print_("Test Losses: {}".format(rounding(self._losses, 4)))
        self.print_("DotNum Losses: {}".format(rounding(self.dotnum, 6)))
        self.print_("Cumulative training time: {}".format(rounding(self._cumulative_times, 2)))
        
    def aggregation(self):
        uploaded_content = self.get_client_uploads()

        list_models = list(uploaded_content[MODEL].values())

        teacher_model = [model for model in list_models]

        weights = list(uploaded_content[DATA_SIZE].values())

        if teacher_model[0] is not None:
            teacher_model = self.aggregate(teacher_model, weights)
            self.set_teacher_model(teacher_model, load_dict=True)

    def set_teacher_model(self, model, load_dict=False):
        if load_dict:
            self.teacher_model.load_state_dict(model.state_dict())
        else:
            self.teacher_model = copy.deepcopy(model)

    def save_model(self):
        """Save the model in the server."""

        if self._do_every(self.conf.server.save_model_every, self._current_round, self.conf.server.rounds) and \
                self.is_primary_server():
            #save_path = self.conf.output
            save_path = self.conf.server.save_model_path
            #pdb.set_trace()
            if save_path == "":
                save_path = os.path.join(os.getcwd(), "saved_models","metafssl_two")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path,
                                     "global_model_r_{}.pth".format(self._current_round))
            torch.save(self._model.cpu().state_dict(), save_path)
            self.print_("Model saved at {}".format(save_path))

    def test(self):
        self.print_("--- start ALL testing ---")
        
        test_begin_time = time.time()
        test_results = {metric.TEST_ACCURACY: 0, metric.TEST_LOSS: 0, metric.TEST_TIME: 0}
        test_results_teacher = {metric.TEST_ACCURACY: 0, metric.TEST_LOSS: 0, metric.TEST_TIME: 0}
        if self.conf.test_mode == TEST_IN_SERVER:
            if self.is_primary_server():
                test_results_teacher = self.test_in_sever_TSModel(self.teacher_model, self.conf.device)
        elif self.conf.test_mode == "test_in_all" or self.conf.test_mode == "weighted_test_in_all":
            if self.is_primary_server():
                test_results_teacher = self.test_in_sever_TSModel(self.teacher_model, self.conf.device)

            ##TODO only sync teacher_model
            self._model = self.teacher_model
            test_results = self.test_in_client()
            test_results[metric.TEST_TIME] = time.time() - test_begin_time
        else:
            test_results = self.test_in_client()
            test_results[metric.TEST_TIME] = time.time() - test_begin_time
        # test_results[metric.TEST_TIME] = time.time() - test_begin_time

        self.track_TStest_results(test_results_teacher, flag='teacher')
        #self.track_test_results(test_results)


    def test_in_sever_TSModel(self, model, device=CPU):
        model.eval()
        model.to(device)
        test_loss = 0
        correct = 0
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        features,labels=None, []
        with torch.no_grad():
            for batched_x, batched_y in self.test_data.loader(self.conf.server.batch_size, seed=self.conf.seed):
                x = batched_x.to(device)
                y = batched_y.to(device)

                x_features,log_probs = model(x,test=True)
                if (self._current_round + 1) == self.conf.server.rounds:
                    if features == None:
                        features = x_features
                    else:
                        features = torch.cat([features, x_features], dim=0)
                
                [labels.append(y[index].item()) for index in range(y.shape[0])]
                
                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                test_loss += loss.item()
            test_data_size = self.test_data.size()
            test_loss /= test_data_size
            accuracy = 100.00 * correct / test_data_size
            if (self._current_round + 1) == self.conf.server.rounds:
               features_array = features.cpu().numpy()
               labels_array=np.array(labels)
               
            del features, labels



            test_results = {
                metric.TEST_ACCURACY: float(accuracy),
                metric.TEST_LOSS: float(test_loss)
            }
            return test_results

    def track_TStest_results(self, results, flag):
        self._cumulative_times.append(time.time() - self._start_time)

        self._losses.append(results[metric.TEST_LOSS])
        if results[metric.TEST_ACCURACY]>0:
            self._accuracies.append(results[metric.TEST_ACCURACY])
        if flag == 'teacher':
            self.print_('TEACHER####Test loss: {:.8f}, Test accuracy: {:.2f}%'.format(
                results[metric.TEST_LOSS], results[metric.TEST_ACCURACY]))
        elif flag == 'student':
            self.print_('STUDENT####Test loss: {:.2f}, Test accuracy: {:.2f}%'.format(
                results[metric.TEST_LOSS], results[metric.TEST_ACCURACY]))

    def distribution_to_test_locally(self):
        """Conduct testing sequentially for selected testing clients."""
        uploaded_accuracies = []
        uploaded_losses = []
        uploaded_data_sizes = []
        uploaded_metrics = []

        test_clients = self.get_test_clients()

        for client in test_clients:
            # Update client config before testing
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self._current_round
            self.conf.client.is_distributed = self.conf.gpu > 0

            uploaded_request = client.run_test(self.teacher_model, self.conf.client)
            uploaded_content = uploaded_request.content
            performance = codec.unmarshal(uploaded_content.data)

            uploaded_accuracies.append(performance.accuracy)
            uploaded_losses.append(performance.loss)
            uploaded_data_sizes.append(uploaded_content.data_size)
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

        self.set_client_uploads_test(uploaded_accuracies, uploaded_losses, uploaded_data_sizes, uploaded_metrics)
