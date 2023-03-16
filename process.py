from models import AlexNet, VGG, RestNet
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
from PIL import Image
import os
import psutil
import json
from networking import *
import argparse
from models import AlexNet, VGG, RestNet


parser = argparse.ArgumentParser(description='Please specify the directory of data set')
parser.add_argument('--port', type=int, default=8005, help="the directory which contains the pictures set.")
parser.add_argument('--ip', type=str, default='146.95.252.28', help="the directory which contains the pictures set.")
args = parser.parse_args()

models = {
    "AlexNet": AlexNet(),
    "VGG": VGG(),
    "RestNet": RestNet()
}

TCP_IP1 = '146.95.252.28'
TCP_PORT1 = args.port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((TCP_IP1, TCP_PORT1))
s.listen(1)
conn, addr = s.accept()
print("client connected")

remote_t = []
model_name = None

try:
    for i in range(8):
        """
        Get data from client
        """
        data = recv_msg(conn)

        """
        Send ack
        """
        ack = np.array([1])
        data_obj = pickle.dumps(ack, protocol=pickle.HIGHEST_PROTOCOL)
        send_msg(conn, data_obj)

        """
        Start server inference
        """
        inputs_ten = pickle.loads(data)
        inputs = torch.from_numpy(inputs_ten[1])
        start_point = inputs_ten[0]
        if i == 0:
            print("start_point=", start_point)
        models[inputs_ten[2]](inputs, None, exe_type=1, server_start_point=start_point,
                              end_point=models[inputs_ten[2]].number_of_layers, data_test=False)
        remote_t.append(models[inputs_ten[2]].local_computation_time[-1])
        models[inputs_ten[2]].local_computation_time = []

        """
        Send server computation time
        """
        ack = np.array([remote_t[-1]])
        data_obj = pickle.dumps(ack, protocol=pickle.HIGHEST_PROTOCOL)
        send_msg(conn, data_obj)

        time.sleep(0.24)
except:
    conn.close()
    s.close()

conn.close()
s.close()




