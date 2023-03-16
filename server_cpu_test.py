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

images = Image.open("9.jpg")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(images)
input_batch = input_tensor.unsqueeze(0)


al = VGG()
t = []
t_layer = [0 for i in range(al.number_of_layers)]
for i in range(300):
    al(input_batch, None, exe_type=0, server_start_point=0, end_point=al.number_of_layers, data_test=True)
    t.append(al.local_computation_time[-1])
    for j in range(al.number_of_layers):
        t_layer[j] += al.layer_latency[j]
    al.local_computation_time = []
    al.layer_latency = []
    al.layer_output_data_size = []

for i in range(al.number_of_layers):
    t_layer[i] = round(t_layer[i]/300, 4)

print(round(np.average(t), 1))
print("t_layer=", np.sum(t_layer), t_layer)

"""
with open("time.json", "r") as jsonFile:
    time_file = json.load(jsonFile)

time_file["alex"].append(t)

with open("time.json", 'w', encoding='utf-8') as f:
    json.dump(time_file, f, indent=4)
"""