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

    nEpochs = 1
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

    # エラー推移
    result = {}
    result["log_loss_G_sum"] = []
    result["log_loss_G_bce"] = []
    result["log_loss_G_mae"] = []
    result["log_loss_D"] = []

    # 訓練
    transform = transforms.Compose( [transforms.ToTensor(),
                                     transforms.Normalize( (0.5,), (0.5,) ) ] )
    dataset_dir = "./test_img"
    print(f"dataset_dir: {dataset_dir}")

    dataset = PairImges(dataset_dir, transform=transform)

    batch_size = 1
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True )

    nBatches = len( trainloader )
    print( 'in train' )

    for i in tqdm(range(nEpochs)):
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

            # 偽画像を本物と騙せるようにロスを計算
            LAMBD = 100.0 # BCEとMAEの係数
            cat_img = torch.cat([fake_img, ori_img], dim=1)
            out = model_D(cat_img)
            # ones_listの最初からbatch_len番目までを取得
            in_ones = ones[:batch_len]
            loss_G_bce = bce_loss(out, in_ones)
            loss_G_mae = LAMBD * mae_loss(fake_img, ans_img)
            loss_G_sum = loss_G_bce + loss_G_mae

            log_loss_G_bce.append(loss_G_bce.item())
            log_loss_G_mae.append(loss_G_mae.item())
            log_loss_G_sum.append(loss_G_sum.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_G_sum.backward()
            params_G.step()

            # Discriminatoの訓練
            # 本物のカラー画像を本物と識別できるようにロスを計算
            real_out = model_D(torch.cat([ans_img, ori_img], dim=1))
            loss_D_real = bce_loss(real_out, ones[:batch_len])

            # 偽の画像の偽と識別できるようにロスを計算
            fake_out = model_D(torch.cat([fake_img_tensor, ori_img], dim=1))
            loss_D_fake = bce_loss(fake_out, zeros[:batch_len])

            # 実画像と偽画像のロスを合計
            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_D.backward()
            params_D.step()

        output.append((fake_img.cpu(), ori_img.cpu()))

        result["log_loss_G_sum"].append(statistics.mean(log_loss_G_sum))
        result["log_loss_G_bce"].append(statistics.mean(log_loss_G_bce))
        result["log_loss_G_mae"].append(statistics.mean(log_loss_G_mae))
        result["log_loss_D"].append(statistics.mean(log_loss_D))
        print(f"log_loss_G_sum = {result['log_loss_G_sum'][-1]} " +
              f"({result['log_loss_G_bce'][-1]}, {result['log_loss_G_mae'][-1]}) " +
              f"log_loss_D = {result['log_loss_D'][-1]}")

    print( 'finished' )

    ####### Save log #######
    log_file_name = "logs_" + str( nEpochs ).zfill( 5 )
    if not os.path.exists("./"+log_file_name):
        os.mkdir("./"+log_file_name)

    #outputの保存
    filename_output = "Map_output_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    print( 'saving ', filename_output )
    torch.save( output, f"./"+log_file_name+"/"+filename_output )

    if not os.path.exists("./"+log_file_name+"/losses"):
        os.mkdir("./"+log_file_name+"/losses")

    #loss_G_sumの保存
    filename_loss_G_sum = "Map_loss_G_sum_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    print( 'saving ', filename_loss_G_sum )
    torch.save( log_loss_G_sum, f"./"+log_file_name+f"/losses/"+filename_loss_G_sum )

    # #loss_G_bceの保存
    # filename_loss_G_bce = "Map_loss_G_bce_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_loss_G_bce )
    # torch.save( log_loss_G_bce, f"./"+log_file_name+f"/losses/"+filename_loss_G_bce )
    #
    # #loss_G_maeの保存
    # filename_loss_G_mae = "Map_loss_G_mae_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_loss_G_mae )
    # torch.save( log_loss_G_mae, f"./"+log_file_name+f"/losses/"+filename_loss_G_mae )

    #loss_Dの保存
    filename_loss_D = "Map_loss_D_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    print( 'saving ', filename_loss_D )
    torch.save( log_loss_D, f"./"+log_file_name+f"/losses/"+filename_loss_D )

    if not os.path.exists("./"+log_file_name+"/models"):
            os.mkdir("./"+log_file_name+"/models")

    #model_Gの保存
    filename_model_G = "Map_model_G_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    print( 'saving ', filename_model_G )
    torch.save( model_G.state_dict(), f"./"+log_file_name+f"/models/"+filename_model_G )

    #model_Dの保存
    filename_model_D = "Map_model_D_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    print( 'saving ', filename_model_D )
    torch.save( model_D.state_dict(), f"./"+log_file_name+f"/models/"+filename_model_D )

if __name__ == "__main__":
    train()
