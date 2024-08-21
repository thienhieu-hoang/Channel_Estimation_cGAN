import h5py
import torch
import torch.nn as nn
import random


import os

FILE_PATH = os.path.dirname(os.path.abspath("__file__"))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SNR = 0 # dB

BATCH_SIZE = 32 #64  # Batch size
NUM_EPOCHS = 20
    
learning_rate = 0.0001

# file path
outer_file_path = os.path.join(FILE_PATH, '..', 'DeepMIMOv2', 'Gan_Data', 'Static_612x14', 'freq_symb_1ant_612sub')
