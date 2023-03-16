import math
import random
import subprocess
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
from networking import *
from models import AlexNet, VGG, RestNet
from _thread import *
from allocation import select_cut_point, do_cpu_allocation, do_b_allocation, cpu_transfer, do_cpu_tx2, paras_dnn_data

cpus = {
    "1": "0",
    "2": "0,1",
    "3": "0,1,2",
    "5": "0,1,2,3,4",
    "7": "0,1,2,3,4,5,6",
    "9": "0,1,2,3,4,5,6,7,8",
    "11": "0,1,2,3,4,5,6,7,8,9,10",
    "13": "0,1,2,3,4,5,6,7,8,9,10,11,12"
}


ports = {
    "AlexNet": "8005",
    "VGG": "8006",
    "RestNet": "8007"
}

cpu_allocation = {
    "AlexNet": "0,1,2",
    "VGG": "4,5,6,7,12,13,14,15",
    "RestNet": "8,9,10,11,3"
}

local_computation_time = {
    "AlexNet": [],
    "VGG": [],
    "RestNet": []
}

network_delay = {
    "AlexNet": [],
    "VGG": [],
    "RestNet": []
}

remote_computation_time = {
    "AlexNet": [],
    "VGG": [],
    "RestNet": []
}

total = {
    "AlexNet": [],
    "VGG": [],
    "RestNet": []
}

pow_hist = {
    "AlexNet": [],
    "VGG": [],
    "RestNet": []
}

finished = 0
global_update = 0
number_of_tx2 = 3
v = 0.00075
q = [0, 0, 0]
cut_point = [0, 0, 0]
types = ["AlexNet", "RestNet", "VGG"]
tx2_tx_pow = [0.5, 0.5, 0.5]
server_feq = 16
tx2_feq = [345600/math.pow(10, 6), 345600/math.pow(10, 6), 345600/math.pow(10, 6)]
server_bandwidth = 12
deadline = [400, 600, 750]
c_allocation = [4, 4, 8]
b_allocation = [8, 8, 8]


def show():
    print("###################")
    print(f"Updates={global_update}")
    print(f"queue = {q}")
    for model_name in ["AlexNet", "VGG", "RestNet"]:
        if len(local_computation_time[model_name]) == 0:
            continue
        print("###################")
        print(f"{model_name}")
        print("\t local_computation_time=", local_computation_time[model_name])
        print("\t network_delay=", network_delay[model_name])
        print("\t remote_computation_time=", remote_computation_time[model_name])
        print("\t\t total=", total[model_name])
        print("\t pow=", pow_hist[model_name])
        print("###################")


def alg(updates):
    global global_update, finished, cut_point, c_allocation, b_allocation, cpu_allocation, tx2_feq
    while True:
        for tx2_id in range(number_of_tx2):
            q[tx2_id] += deadline[tx2_id] / 1000.
        #cut_point = select_cut_point(number_of_tx2, v, q, cut_point, types, tx2_tx_pow, tx2_feq, server_feq,
        #                             server_bandwidth,
        #                             deadline)
        c_allocation = do_cpu_allocation(number_of_tx2, q, cut_point, types, server_feq)
        b_allocation = do_b_allocation(number_of_tx2, v, q, cut_point, types, tx2_tx_pow, server_bandwidth)
        tx2_feq = do_cpu_tx2(number_of_tx2, q, v, cut_point)
        cpu_number = cpu_transfer(c_allocation, server_feq)
        print(cpu_number, cut_point, tx2_feq, b_allocation)
        cpu_allocation = {
            "AlexNet": cpu_number[0],
            "RestNet": cpu_number[1],
            "VGG": cpu_number[2]
        }

        while finished != 3:
            time.sleep(2)
        for tx2_id in range(number_of_tx2):
            # network_delay[types[tx2_id]][-1] = 1000 * paras_dnn_data[types[tx2_id]][cut_point[tx2_id]]/b_allocation[tx2_id]
            # total[types[tx2_id]][-1] = network_delay[types[tx2_id]][-1] + local_computation_time[types[tx2_id]][-1] + remote_computation_time[types[tx2_id]][-1]
            q[tx2_id] = max(0, q[tx2_id] - deadline[tx2_id]/1000. + (total[types[tx2_id]][-1] - deadline[tx2_id]) / 1000.)
        show()
        time.sleep(1)
        global_update += 1
        finished = 0


def user_controller(user_conn):
    global global_update, finished
    local_update = 0
    cut = 0
    while cut <= 39:
        data = recv_msg(user_conn)
        model_name = pickle.loads(data)[0]
        """
            Send cut_point to user
            """
        INDEX = 0
        if model_name == "AlexNet":
            INDEX = 0
        elif model_name == "RestNet":
            INDEX = 1
        elif model_name == "VGG":
            INDEX = 2

        #ack = np.array([cut_point[INDEX], tx2_feq[INDEX] * math.pow(10, 6), b_allocation[INDEX] * 1000])
        ack = np.array([cut, 1881600, 20480])
        data_obj = pickle.dumps(ack, protocol=pickle.HIGHEST_PROTOCOL)
        send_msg(user_conn, data_obj)
        print("start ", model_name, f"with cut-point {cut_point[INDEX]} and cpus {cpu_allocation[model_name]}")
        p = subprocess.Popen(["taskset", "-c", cpu_allocation[model_name], "python3", "process.py", "--port", ports[model_name]])
        p.wait()

        data = recv_msg(user_conn)
        times = pickle.loads(data)
        local_computation_time[model_name].append(times[0])
        network_delay[model_name].append(times[1])
        remote_computation_time[model_name].append(times[2])
        pow_hist[model_name].append(times[3])
        total[model_name].append(round(times[0] + times[1] + times[2], 4))
        local_update += 1
        finished += 1

        show()

        cut += 1
        #while local_update > global_update:
        #    time.sleep(1)


#start_new_thread(alg, (100,))

TCP_IP1 = '146.95.252.28'
TCP_PORT1 = 8002

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((TCP_IP1, TCP_PORT1))
s.listen(3)
user_s = []
while True:
    try:
        conn, addr = s.accept()
        print(f"{addr}-user connected")
        user_s.append(conn)
        start_new_thread(user_controller, (conn,))
    except:
        break



