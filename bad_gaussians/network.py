import torch
import torch.nn as nn
import numpy as np
import cv2

def conv_layer(inDim, outDim, ks, s, p, norm_layer='none'):
    ## convolutional layer
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.ReLU(True)
    assert norm_layer in ('batch', 'instance', 'none')
    if norm_layer == 'none':
        seq = nn.Sequential(*[conv, relu])
    else:
        if (norm_layer == 'instance'):
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False) # instance norm
        else:
            momentum = 0.1
            norm = nn.BatchNorm2d(outDim, momentum = momentum, affine=True, track_running_stats=True)
        seq = nn.Sequential(*[conv, norm, relu])
    return seq

class ResBlock(nn.Module):
    def __init__(self, channels=128):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        x2 = self.linear2(x1)
        return x + x2
    
class SpkRecon_Net(nn.Module):
    def __init__(self, input_dim=41):
        super().__init__()
        self.norm = 'none'
        self.out_dim = 1
        
        self.convBlock1 = conv_layer(input_dim, 64, 3, 1, 1)
        self.convBlock2 = conv_layer(64, 128, 3, 1, 1, self.norm)
        self.convBlock3 = conv_layer(128, 64, 3, 1, 1, self.norm)
        self.convBlock4 = conv_layer(64, 16, 3, 1, 1, self.norm)
        self.conv = nn.Conv2d(16, self.out_dim, 3, 1, 1)
        
        self.seq = nn.Sequential(
            self.convBlock1, 
            self.convBlock2, 
            self.convBlock3, 
            self.convBlock4, 
            self.conv
        )
    
    def forward(self, spike):
        return self.seq(spike)
    
class Multi_SpkRecon_Net(nn.Module):
    def __init__(self, input_dim=41,voxel_dim = 27,max_position = 13):
        super().__init__()
        self.norm = 'none'
        self.out_dim = 1
        embedding_dim = 64
        self.preprocess_spk = conv_layer(input_dim, embedding_dim, 3, 1, 1)
        self.preprocess_voxel = conv_layer(voxel_dim,embedding_dim,3,1,1)
        self.convBlock1 = conv_layer(64, 128, 3, 1, 1, self.norm)
        self.convBlock2 = conv_layer(128, 64, 3, 1, 1, self.norm)
        self.convBlock3 = conv_layer(64, 16, 3, 1, 1, self.norm)
        self.conv = nn.Conv2d(16, self.out_dim, 3, 1, 1)

        self.seq = nn.Sequential(
            self.convBlock1, 
            self.convBlock2, 
            self.convBlock3, 
            self.conv
        )

    def forward(self,voxel,spike,index):
        voxel = self.preprocess_voxel(voxel)
        spike = self.preprocess_spk(spike)
        index = index * torch.ones_like(voxel)
        feature = voxel + spike + index
        return self.seq(feature)

if __name__ == "__main__":
    recon_net = Multi_SpkRecon_Net()
    spike = torch.zeros((1,41,256,256))
    voxel = torch.zeros((1,27,256,256))
    print(recon_net(voxel,spike,index = torch.tensor([0.5]).cuda()).shape)
    