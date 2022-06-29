import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from tqdm import tqdm
import statistics

from PIL import Image
import glob

import os
from os import listdir

from UNetGenerator import UNetGenerator
from Discriminator import MyDiscriminator, Discriminator
from UNet_dataset  import PairImges

nEpochs = 200
print("Enter the number of epochs you want to animate (default is 200)")
nEpochs = input("")

log_file_name = "logs_" + str( nEpochs ).zfill( 5 )

print(f"log_file_name: {log_file_name}")

filename_output = "Map_output_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
print( 'loading ', filename_output )
output_and_label = torch.load( f"./"+log_file_name+"/"+filename_output, map_location=device )
#
filename_model_G = "Map_model_G_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
print( 'loading ', filename_model_G )
model_G.load_state_dict( torch.load( f"./"+log_file_name+f"/models/"+filename_model_G, map_location=device ) )
#
filename_model_D = "Map_model_G_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
print( 'loading ', filename_model_D )
model_D.load_state_dict( torch.load( f"./"+log_file_name+f"/models/"+filename_model_D, map_location=device ) )
#
model_G.eval()
model_D.eval()
