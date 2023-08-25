import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
#from utils import *
#from utils import homo_warp
from inplace_abn import InPlaceABN
#from renderer import run_network_mvs


#############################################     MVS Net models        ################################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))

class FeatureAngle(nn.Module):
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureAngle, self).__init__()


      
        self.convf = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))
        self.conva = nn.Sequential(
                        ConvBnReLU(12, 8, 3, 1, 1, norm_act=norm_act))

    def forward(self, feature, angle):
        # x: (B, 3, H, W)
        feature = self.convf(feature) # (B, 8, H, W)

        angle = self.conva(angle) # (B, 8, H, W)

        feature = torch.cat([feature,angle],1)
        del angle

        return feature
        
###################################  feature net  ######################################
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()


      
        self.conv0 = nn.Sequential(
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(32, 64, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(64, 64, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(64, 64, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(64, 64, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        #print("x shape:",x.shape)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)

        x = self.toplayer(x) # (B, 32, H//4, W//4)

        return x



class Feature_2dto3d(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(Feature_2dto3d, self).__init__()
      
        self.conv0 = nn.Sequential(
                        ConvBnReLU(256, 512, 1, 1,0,norm_act=norm_act),
                        ConvBnReLU(512, 1024, 1, 1,0, norm_act=norm_act),
                        ConvBnReLU(1024, 2048, 1,1,0, norm_act=norm_act))
        self.toplayer = nn.Conv2d(2048, 2048, 1)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        #print("x shape:",x.shape)
        x = self.toplayer(x) # (B, 32, H//4, W//4)
        return x.view(x.shape[0], 32,64, x.shape[2],x.shape[3])


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 16, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(64, 128, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(128, 128, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv12 = nn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        #print("conv0", conv0.shape)   #conv0 torch.Size([3, 8, 64, 75, 100])
                                      #conv1 torch.Size([3, 16, 32, 38, 50])
        conv2 = self.conv2(self.conv1(conv0))  #conv2 torch.Size([3, 16, 32, 38, 50])

        #print("conv1:", self.conv1(conv0).shape) 
        conv4 = self.conv4(self.conv3(conv2))
        #print("conv4:", conv4.shape)
        #print("self.conv5(x)",self.conv5(conv4).shape)
        x = self.conv6(self.conv5(conv4))
        #print("conv6:",x.shape)
        #print("self.conv7(x)",self.conv7(x).shape)
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.conv12(x)
        return x


