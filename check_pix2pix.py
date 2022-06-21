# https://atmarkit.itmedia.co.jp/ait/articles/2007/10/news024_2.html
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

from sklearn import manifold
from sklearn.manifold import TSNE

import os
from os import listdir

from UNetGenerator import UNetGenerator
from Discriminator import MyDiscriminator, Discriminator
from UNet_dataset import PairImges

# Dimensinality reduction: t-SNE
FIXED_SEED = 3
# before training/inference:
np.random.seed( FIXED_SEED )

def tsne( X ):
    X_dataset = np.array( X )
    # print( " X_dataset = ", X_dataset );
    #tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 30.0)
    #tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 5.0)

    # changed for the new version by S.T. on 2021/04/07
    #tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 5.0, random_state = 3 )
    # tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 5.0, random_state = 3, square_distances=True )
    tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 30.0, random_state = 3, square_distances=True )

    #tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 4.0)
    Y_dataset = tSNE.fit_transform( X_dataset )
    Y_dataset = Y_dataset.astype(np.float64)
    print("t-SNE Dimensionality Reduction Done.")
    #return Y_dataset.data.numpy()
    return Y_dataset


# main program starts
#input size
inSize = 256

# モデル
device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
torch.backends.cudnn.benchmark = True

nEpochs = 1000
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

#with open( 'output_and_label.pickle', mode='rb' ) as f:
#    output_and_label = pickle.load( f )
#with open( 'losses.pickle', mode='rb' ) as f:
#    losses = pickle.load( f )
#model.load_state_dict( torch.load( 'MNIST_model.pth' ) )

log_file_name = "logs_" + str( nEpochs ).zfill( 5 )

filename_output = "Map_output_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
print( 'loading ', filename_output )
output_and_label = torch.load( f"./"+log_file_name+"/"+filename_output, map_location=device )
#
filename_loss_G_sum = "Map_loss_G_sum_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
print( 'loading ', filename_loss_G_sum)
losse_G_sum = torch.load( f"./"+log_file_name+f"/losses/"+filename_loss_G_sum, map_location=device )
#
filename_loss_D = "Map_loss_D_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
print( 'loading ', filename_loss_D)
losse_G_sum = torch.load( f"./"+log_file_name+f"/losses/"+filename_loss_D, map_location=device )
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

# define the function to preview the image
def imshow( img ):
    img = torchvision.utils.make_grid( img, padding=16 )
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow( np.transpose( npimg, (1, 2, 0) ) )
    plt.show()


# show how the set of images get better at every two epochs
# output_and_label[ 0 : 10 : 2 ] <= From epoch 0 to 9 by sampling every 2 epochs
step = int( nEpochs / 10 )
#step = int( nEpochs / 5 )
#step = 1
for i, ( imgP, imgO ) in enumerate( output_and_label[ 0 : nEpochs : step ] ):
    print( ' set of original images at epoch: ', i*step )
    imshow( imgO.detach().reshape( -1, 1, inSize, inSize ).cpu() )
    print( ' set of predicted images at epoch: ', i*step )
    imshow( imgP.detach().reshape( -1, 1, inSize, inSize ).cpu() )

# show the set of predicted and original images at the last epoch
pred, orig = output_and_label[ -1 ]
#print( ' pred.size() = ', pred.size() )
#print( ' orig.size() = ', orig.size() )
print( ' original image ' )
imshow( orig.reshape( -1, 1, inSize, inSize ).cpu() )
print( ' predicted image ' )
imshow( pred.reshape( -1, 1, inSize, inSize ).cpu() )

# iterator = iter( testloader )
# img, _ = next( iterator )
# img = img.reshape( -1, 28 * 28 )
# output = model( img )
# imshow( img.reshape(-1, 1, 28, 28) )
# imshow( output.reshape(-1, 1, 28, 28) )

sys.exit()

sampleloader = DataLoader( testset, batch_size=1500 )
iterator = iter( sampleloader )
orig, pred = next( iterator )
#z = model.enc( orig.reshape(-1, 28 * 28) )
z = model.enc( orig )
z = z.detach().numpy()  # 後から簡単に使えるようにするための処理
print( z.shape )  # (1500, 2)
z.reshape( 1500, 32*7*7 )
y = tsne( z )
print( y.shape )  # (1500, 2)

nDigits = 10
set_list = [ set() for x in range( nDigits ) ]
for coord, lbl in zip( y.tolist(), pred ):
    set_list[ lbl ].add( tuple( coord ) )

for idx in range( nDigits ):
    print( f'items in set_list[{idx}]:' )
    for cnt, item in enumerate( set_list[ idx ] ):
        # print( ' cnt = ', cnt )
        # print( ' type( item ) = ', type( item ) )
        # print( ' len( item ) = ', len( item ) )
        print( item )
        if cnt > 5:
            break

colorlist = [ "r", "g", "b", "c", "k", "y", "orange", "lightgreen", "hotpink", "yellow" ]
plt.figure( figsize=(10, 10) )
for idx in range( nDigits ):
    for x, y in set_list[ idx ]:
        plt.scatter( x, y, c=colorlist[ idx ] )
description = [ f"{idx}: {colorlist[idx]}" for idx in range(10) ]
print( description )
plt.show()
