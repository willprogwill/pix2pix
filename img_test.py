import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
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
from UNet_dataset import PairImges

def train():
    #input size
    inSize = 256

    # モデル
    device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
    torch.backends.cudnn.benchmark = True

    nEpochs = 3
    args = sys.argv
    if len( args ) == 2:
        nEpochs = int(args[ 1 ] )

    print( ' nEpochs = ', nEpochs )
    print( ' device = ', device )

    model_G, model_D = UNetGenerator(), MyDiscriminator()
    model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)
    model_G, model_D = model_G.to(device), model_D.to(device)

    params_G = torch.optim.Adam(model_G.parameters(),
                lr=0.0002, betas=(0.5, 0.999))
    params_D = torch.optim.Adam(model_D.parameters(),
                lr=0.0002, betas=(0.5, 0.999))

    # ロスを計算するためのラベル変数 (PatchGAN)
    ones = torch.ones(32, 1, 4, 4).to(device)
    zeros = torch.zeros(32, 1, 4, 4).to(device)

    # ロスを計算するためのラベル変数 (DCGAN)
    # ones = torch.ones(32).to(device)
    # zeros = torch.zeros(32).to(device)

    # 損失関数
    bce_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()

    # 訓練
    transform = transforms.Compose( [transforms.ToTensor(),
                                     transforms.Normalize( (0.5,), (0.5,) ) ] )
    dataset_dir = "./half"
    print(f"dataset_dir: {dataset_dir}")

    dataset = PairImges(dataset_dir, transform=transform)

    batch_size = 32
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False )

    nBatches = len( trainloader )
    print( 'in train' )

    for i in range(nEpochs):
        log_loss_G_sum, log_loss_G_bce, log_loss_G_mae, log_loss_D = [], [], [], []
        output =[]

        #for counter, (ans_img, ori_img) in enumerate(trainloader):
        for ans_img, ori_img in trainloader:

            #print( counter, ' / ', nBatches )

            batch_len = len(ans_img)
            ans_img, ori_img = ans_img.to(device), ori_img.to(device)

            # Gの訓練
            # 偽画像を作成
            fake_img = model_G(ori_img)
            # 偽画像を一時保存
            fake_img_tensor = fake_img.detach()

            break

        # 画像を保存
        if not os.path.exists("imgs"):
            os.mkdir("imgs")

        # 生成画像を保存
        torchvision.utils.save_image(fake_img_tensor[:min(batch_len, 100)],
                                     f"imgs/fake_epoch_{i:03}.png",
                                     value_range=(-1.0, 1.0), normalize=True)
        torchvision.utils.save_image(ans_img[:min(batch_len, 100)],
                                     f"imgs/real_epoch_{i:03}.png",
                                     value_range=(-1.0, 1.0), normalize=True)


    print( '====== finished ======' )

    # 最後の生成画像を保存
    torchvision.utils.save_image(fake_img_tensor[:min(batch_len, 100)],
                                 f"imgs/fake_epoch_{i:03}.png",
                                 value_range=(-1.0,1.0), normalize=True)
    torchvision.utils.save_image(ans_img[:min(batch_len, 100)],
                                 f"imgs/real_epoch_{i:03}.png",
                                 value_range=(-1.0, 1.0), normalize=True)

if __name__ == "__main__":
    train()
