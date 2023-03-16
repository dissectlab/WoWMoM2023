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
from networking import *
import argparse
from models import AlexNet, VGG, RestNet
from multiprocessing.pool import ThreadPool
from _thread import start_new_thread

parser = argparse.ArgumentParser(description='Please specify the directory of data set')
parser.add_argument('--model', type=str, default="AlexNet", help="the directory which contains the pictures set.")
parser.add_argument('--port', type=int, default=8005, help="the directory which contains the pictures set.")
parser.add_argument('--ip', type=str, default='192.168.1.152', help="the directory which contains the pictures set.")
parser.add_argument('--feq', default=1881600, type=int, help='an integer for the accumulator')
args = parser.parse_args()


pow_hist = []


def batter_monitor():
    while True:
        pow = subprocess.check_output(
            "echo 123456 | sudo cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input",
            shell=True)
        pow_hist.append(int(pow))


def run():
    images = Image.open("9.jpg")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(images)
    input_batch = input_tensor.unsqueeze(0)

    print("####################################################")
    if args.model == "AlexNet":
        model1 = AlexNet()
    elif args.model == "VGG":
        model1 = VGG()
    elif args.model == "RestNet":
        model1 = RestNet()

    # connect to controller
    TCP_IP2 = args.ip
    TCP_PORT2 = 8002
    controller = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    controller.connect((TCP_IP2, TCP_PORT2))

    process = None

    for i in range(model1.number_of_layers+1):
        y = np.array([args.model])
        data_obj = pickle.dumps(y, protocol=pickle.HIGHEST_PROTOCOL)
        send_msg(controller, data_obj)
        print(f"{args.model} is on, connect to controller, wait for ACK")

        while True:
            data = recv_msg(controller)
            break

        if process is None:
            # connect to server process
            TCP_IP2 = args.ip
            TCP_PORT2 = args.port
            while True:
                try:
                    process = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    process.connect((TCP_IP2, TCP_PORT2))
                    break
                except:
                    time.sleep(2)
            print(f"Connect to server on port {TCP_PORT2}")

        pow_json = {"hist": []}
        with open("pow.json", 'w', encoding='utf-8') as f:
            json.dump(pow_json, f, indent=4)
        p = subprocess.Popen(["python3", "pow2.py"])

        rc = subprocess.Popen(["sudo", "./cpu.sh", str(args.feq)])
        rc.wait()

        for j in range(10):
            model1(input_batch, process, exe_type=1, server_start_point=0, end_point=i, data_test=False)
            print(j)

        p.kill()

        with open("pow.json", "r") as jsonFile:
            pow_hist = json.load(jsonFile)["hist"]

        if len(model1.local_computation_time) == 0:
            model1.local_computation_time.append(0)
        if len(model1.network_delay) == 0:
            model1.network_delay.append(0)
        if len(model1.remote_computation_time) == 0:
            model1.remote_computation_time.append(0)

        y = np.array([round(np.average(model1.local_computation_time[1:]), 4),
                      round(np.average(model1.network_delay[1:]), 4),
                      round(np.average(model1.remote_computation_time[1:]), 4), round(np.average(pow_hist), 4)])
        print(y)

        data_obj = pickle.dumps(y, protocol=pickle.HIGHEST_PROTOCOL)
        send_msg(controller, data_obj)

        model1.local_computation_time = []
        model1.network_delay = []
        model1.remote_computation_time = []

        process.close()
        process = None

        time.sleep(4)


run()