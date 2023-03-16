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

original_model = models.resnet18(pretrained=True)

print("Main model")
child_counter = 0
for child in original_model.children():
    print(" child", child_counter, "is:")
    print(child)
    child_counter += 1