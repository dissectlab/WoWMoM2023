import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import sys
import cv2
import socket
import pickle
import json
from timeit import default_timer as timer
from networking import *


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(AlexNet, self).__init__()
        self.layers = [
            ##########################
            # features
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True)
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True)
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            [
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                #############################
                nn.AdaptiveAvgPool2d((6, 6)),
                #############################
                # classifier
                nn.Dropout()
            ],
            nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout()
            ),
            nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True)
            ),
            nn.Linear(4096, num_classes)
        ]
        self.number_of_layers = len(self.layers)

        # the accumulated computation time till to current layer
        self.remote_computation_time = []
        self.local_computation_time = []
        # the data size of this layer's output
        self.layer_output_data_size = []
        # the computation latency of current layer
        self.layer_latency = []
        #
        self.network_delay = []

    def forward(self, x, connect, exe_type=0, server_start_point=0, end_point=0, data_test=False):
        for layer_ID in range(server_start_point):
            self.local_computation_time.append(0)
            self.remote_computation_time.append(0)
            self.layer_output_data_size.append(0)
            self.layer_latency.append(0)

        """
        Server do layers [server_start_point, number_of_layers]
        """
        dnn_start_time = time.time()
        for layer_ID in range(server_start_point, end_point):
            layer_start_time = timer()

            """
            Handle data of "AdaptiveAvgPool2d"
            """
            if layer_ID == 6:
                for l in range(len(self.layers[layer_ID])):
                    x = self.layers[layer_ID][l](x)
                    if l == 3:
                        x = torch.flatten(x, 1)
            else:
                x = self.layers[layer_ID](x)
            """
            Measure time
            """
            self.layer_latency.append(round((timer() - layer_start_time) * 1000, 4))
            # save that data size of this layer's output
            """
            Measure data size
            """
            if data_test:
                data_obj = pickle.dumps(x.detach().numpy(), protocol=pickle.HIGHEST_PROTOCOL)
                self.layer_output_data_size.append(round(sys.getsizeof(data_obj) / (1024 * 1024), 4))

        self.local_computation_time.append(round((time.time() - dnn_start_time) * 1000, 4))

        if exe_type == 1 and end_point != self.number_of_layers:
            t1 = time.time()
            y = np.array([end_point, x.detach().numpy(), "AlexNet"])
            data_obj = pickle.dumps(y, protocol=pickle.HIGHEST_PROTOCOL)
            send_msg(connect, data_obj)
            """
            Measure network delay
            """
            # print("Data is sent, wait for ACK")
            while True:
                data = recv_msg(connect)
                break
            self.network_delay.append(round((time.time() - t1) * 1000, 4))
            """
            Measure server computation time
            """
            while True:
                data = recv_msg(connect)
                inputs_ten = pickle.loads(data)
                self.remote_computation_time.append(inputs_ten[0])
                break
        return x


class VGG(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(VGG, self).__init__()
        self.layers = [
            ##########################
            # features
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 1
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 2
                nn.ReLU(inplace=True)
            ),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 3

            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 4
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 5
                nn.ReLU(inplace=True)
            ),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 6

            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 7
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 8
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 9
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            ),

            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 10
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 11
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 12
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ),

            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.ReLU(inplace=True)
            ),

            [
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.AdaptiveAvgPool2d(output_size=(7, 7)),  # 31
            ],

            nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False)
            ),

            nn.Sequential(
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False)
            ),

            nn.Linear(in_features=4096, out_features=1000, bias=True)
        ]
        self.number_of_layers = len(self.layers)

        # the accumulated computation time till to current layer
        self.remote_computation_time = []
        self.local_computation_time = []
        # the data size of this layer's output
        self.layer_output_data_size = []
        # the computation latency of current layer
        self.layer_latency = []
        #
        self.network_delay = []

    def forward(self, x, connect, exe_type=0, server_start_point=0, end_point=0, data_test=False):
        for layer_ID in range(server_start_point):
            self.local_computation_time.append(0)
            self.remote_computation_time.append(0)
            self.layer_output_data_size.append(0)
            self.layer_latency.append(0)

        """
        Server do layers [server_start_point, number_of_layers]
        """
        dnn_start_time = time.time()
        for layer_ID in range(server_start_point, end_point):
            layer_start_time = timer()

            """
            Handle data of "AdaptiveAvgPool2d"
            """
            if layer_ID == self.number_of_layers - 4:
                for l in range(len(self.layers[layer_ID])):
                    x = self.layers[layer_ID][l](x)
                    if l == 3:
                        x = torch.flatten(x, 1)
            else:
                x = self.layers[layer_ID](x)
            """
            Measure time
            """
            self.layer_latency.append(round((timer() - layer_start_time) * 1000, 4))
            # save that data size of this layer's output
            """
            Measure data size
            """
            if data_test:
                data_obj = pickle.dumps(x.detach().numpy(), protocol=pickle.HIGHEST_PROTOCOL)
                self.layer_output_data_size.append(round(sys.getsizeof(data_obj) / (1024 * 1024), 4))

        self.local_computation_time.append(round((time.time() - dnn_start_time) * 1000, 4))

        if exe_type == 1 and end_point != self.number_of_layers:
            t1 = time.time()
            y = np.array([end_point, x.detach().numpy(), "VGG"])
            data_obj = pickle.dumps(y, protocol=pickle.HIGHEST_PROTOCOL)
            send_msg(connect, data_obj)
            """
            Measure network delay
            """
            # print("Data is sent, wait for ACK")
            while True:
                data = recv_msg(connect)
                break
            self.network_delay.append(round((time.time() - t1) * 1000, 4))
            """
            Measure server computation time
            """
            while True:
                data = recv_msg(connect)
                inputs_ten = pickle.loads(data)
                self.remote_computation_time.append(inputs_ten[0])
                break

        return x


class RestNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(RestNet, self).__init__()
        self.layers = [
            ##########################
            # features
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),         # 1
            nn.Sequential(
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Sequential(
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Sequential(
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Sequential(
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),     # 19
            #nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
            #nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Sequential(
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Sequential(
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),     # 29
            #nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False),
            #nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Sequential(
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Sequential(
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),      # 39
            #nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
            #nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Sequential(
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(in_features=512, out_features=1000, bias=True)
        ]
        self.number_of_layers = len(self.layers)

        # the accumulated computation time till to current layer
        self.remote_computation_time = []
        self.local_computation_time = []
        # the data size of this layer's output
        self.layer_output_data_size = []
        # the computation latency of current layer
        self.layer_latency = []
        #
        self.network_delay = []

    def forward(self, x, connect, exe_type=0, server_start_point=0, end_point=0, data_test=False):
        for layer_ID in range(server_start_point):
            self.local_computation_time.append(0)
            self.remote_computation_time.append(0)
            self.layer_output_data_size.append(0)
            self.layer_latency.append(0)

        """
        Server do layers [server_start_point, number_of_layers]
        """
        dnn_start_time = time.time()
        for layer_ID in range(server_start_point, end_point):
            layer_start_time = timer()
            x = self.layers[layer_ID](x)
            """
            Handle data of "AdaptiveAvgPool2d"
            """
            if layer_ID == self.number_of_layers - 2:
                x = torch.flatten(x, 1)
            """
            Measure time
            """
            self.layer_latency.append(round((timer() - layer_start_time) * 1000, 4))
            # save that data size of this layer's output
            """
            Measure data size
            """
            if data_test:
                data_obj = pickle.dumps(x.detach().numpy(), protocol=pickle.HIGHEST_PROTOCOL)
                self.layer_output_data_size.append(round(sys.getsizeof(data_obj) / (1024 * 1024), 4))

        self.local_computation_time.append(round((time.time() - dnn_start_time) * 1000, 4))

        if exe_type == 1 and end_point != self.number_of_layers:
            t1 = time.time()
            y = np.array([end_point, x.detach().numpy(), "RestNet"])
            data_obj = pickle.dumps(y, protocol=pickle.HIGHEST_PROTOCOL)
            send_msg(connect, data_obj)
            """
            Measure network delay
            """
            # print("Data is sent, wait for ACK")
            while True:
                data = recv_msg(connect)
                break
            self.network_delay.append(round((time.time() - t1) * 1000, 4))
            """
            Measure server computation time
            """
            while True:
                data = recv_msg(connect)
                inputs_ten = pickle.loads(data)
                self.remote_computation_time.append(inputs_ten[0])
                break
        return x
