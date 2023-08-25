#from __future__ import absolute_import, division, print_function
import numpy as np
import math
#from volume_sampler import apply_volume_transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T
import scipy.io as sio


class Conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride, is_3d_conv=False, dilation=1,
                 use_normalization=True,
                 use_relu=False):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.is_3d_conv = is_3d_conv
        self.dilation = dilation
        self.use_normalization = use_normalization
        self.use_relu = use_relu

        if not self.is_3d_conv:
            self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                                       dilation=self.dilation)
            if self.use_normalization:
                self.normalize = nn.BatchNorm2d(num_out_layers)

        else:
            self.conv_base = nn.Conv3d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                                       dilation=self.dilation)
            if self.use_normalization:
                self.normalize = nn.BatchNorm3d(num_out_layers)

    def forward(self, x):
        p = int(np.floor(self.dilation * (self.kernel_size - 1) / 2))
        if not self.is_3d_conv:
            pd = (p, p, p, p)
        else:
            #for 3d features, p==1
            pd = (p, p, p, p, p, p)
        x = self.conv_base(F.pad(x, pd))
        if self.use_normalization:
            x = self.normalize(x)
        if self.use_relu:
            return F.relu(x, inplace=True)
        else:
            return F.elu(x, inplace=True)


class ResConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride, kernel_size=3, is_3d_conv=False):
        super(ResConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = Conv(num_in_layers, num_out_layers, 1, 1, self.is_3d_conv)
        self.conv2 = Conv(num_out_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                          is_3d_conv=self.is_3d_conv)
        if not self.is_3d_conv:
            self.conv3 = nn.Conv2d(num_out_layers, 4 * num_out_layers, kernel_size=1, stride=1)
            self.conv4 = nn.Conv2d(num_in_layers, 4 * num_out_layers, kernel_size=1, stride=stride)
            self.normalize = nn.BatchNorm2d(4 * num_out_layers)
        else:
            self.conv3 = nn.Conv3d(num_out_layers, 4 * num_out_layers, kernel_size=1, stride=1)
            self.conv4 = nn.Conv3d(num_in_layers, 4 * num_out_layers, kernel_size=1, stride=stride)
            self.normalize = nn.BatchNorm3d(4 * num_out_layers)

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        shortcut = self.conv4(x)
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def ResBlock(num_in_layers, num_out_layers, num_blocks, stride, kernel_size=3, is_3d_conv=False):
    layers = [
        ResConv(num_in_layers, num_out_layers, stride, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
    ]

    for i in range(1, num_blocks - 1):
        layers.append(
            ResConv(4 * num_out_layers, num_out_layers, 1, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
        )

    layers.append(
        ResConv(4 * num_out_layers, num_out_layers, 1, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
    )
    return nn.Sequential(*layers)


class UpConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale, is_3d_conv=False):
        super(UpConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.up_nn = nn.Upsample(scale_factor=scale)
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, 1, is_3d_conv=is_3d_conv)

    def forward(self, x):
        x = self.up_nn(x)
        return self.conv1(x)


class OutputConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=3, is_3d_conv=False, kernel_size=3):
        super(OutputConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.kernel_size = kernel_size
        self.sigmoid = torch.nn.Sigmoid()
        if not self.is_3d_conv:
            self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=self.kernel_size, stride=1)
        else:
            self.conv1 = nn.Conv3d(num_in_layers, num_out_layers, kernel_size=self.kernel_size, stride=1)

    def forward(self, x):
        if self.kernel_size > 1:
            p = 1
            if not self.is_3d_conv:
                pd = (p, p, p, p)
            else:
                pd = (p, p, p, p, p, p)
            x = self.conv1(F.pad(x, pd))
        else:
            x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class TBN(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, vol_dim=32, num_features=2048,tensor_type='torch.HalfTensor' ):

        super(TBN, self).__init__()

        #self.args = args
        self.vol_dim = vol_dim
        self.num_features = num_features
        self.tensor_type = tensor_type
        self.num_input_convs = 1
        self.num_gen_features = 128
        self.encode_feature_scale_factor = 1
        self.num_res_convs = 2
        self.num_enc_features = int(self.num_gen_features/self.encode_feature_scale_factor)
        #self.num_dec_features = int(self.args.num_gen_features/self.args.decode_feature_scale_factor)
        
        in_layers = num_in_layers
        if 0 < self.num_input_convs:
            init_num_in_layers = num_in_layers
            middle_num_in_layers = self.num_enc_features
            middle_num_out_layers = middle_num_in_layers
            final_num_out_layers = middle_num_in_layers
            in_layers = final_num_out_layers
            in_conv_layers = []

            for idx in range(self.num_input_convs):
                if 0 == idx:
                    conv_in_layers = init_num_in_layers
                    conv_out_layers = middle_num_in_layers
                elif self.num_input_convs - 1 == idx:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = middle_num_out_layers
                else:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = final_num_out_layers

                in_conv_layers.append(
                    Conv(num_in_layers=conv_in_layers, num_out_layers=conv_out_layers, kernel_size=4, stride=2,
                         is_3d_conv=False, dilation=1, use_normalization=True)
                )

            self.in_conv = nn.Sequential(*in_conv_layers)

        self.conv1_2d_encode = Conv(in_layers, 2 * self.num_enc_features, 2, 2)
        self.conv2_2d_encode = ResBlock(2 * self.num_enc_features, self.num_enc_features, self.num_res_convs, 2)
        self.conv3_2d_encode = ResBlock(4 * self.num_enc_features, 2 * self.num_enc_features, self.num_res_convs, 2)
        self.conv4_2d_encode = ResBlock(8 * self.num_enc_features,  4* self.num_enc_features, self.num_res_convs, 2)
        #self.conv5_2d_encode = ResBlock(16* self.num_enc_features, 8 * self.num_enc_features, self.num_res_convs, 2)

        #self.upconv6_2d_encode = UpConv(64 * self.num_enc_features, 32 * self.num_enc_features, 3, 2)
        #self.iconv6_2d_encode = Conv(2 * 32 * self.num_enc_features, 32 * self.num_enc_features, 3, 1)

        #self.upconv5_2d_encode = UpConv(32 * self.num_enc_features, 16 * self.num_enc_features, 3, 2)
        #self.iconv5_2d_encode = Conv(2 * 16 * self.num_enc_features, 16 * self.num_enc_features, 3, 1)
        
        self.upconv4_2d_encode = UpConv(16 * self.num_enc_features, 8 * self.num_enc_features, 3, 2)
        self.iconv4_2d_encode = Conv(2 * 8 * self.num_enc_features, 8 * self.num_enc_features, 3, 1)

        self.upconv3_2d_encode = UpConv(8 * self.num_enc_features, 4 * self.num_enc_features, 3, 2)
        self.iconv3_2d_encode = Conv(2 * 4 * self.num_enc_features, 4 * self.num_enc_features, 3, 1)

        self.upconv2_2d_encode = UpConv(4 * self.num_enc_features, self.num_features, 3, 2)
        self.iconv2_2d_encode = Conv(2 * self.num_enc_features + self.num_features, self.num_features, 3, 1)
        #self.upconv1_2d_encode = UpConv(4*self.num_enc_features, 512, 3, 2)
        #self.upconv0_2d_encode = UpConv(4*self.num_enc_features, 512, 3, 2)

        #num_3d_features = int(self.num_features / self.vol_dim)
        #self.conv1_3d_encode = Conv(num_3d_features, self.num_enc_features, 3, 1, is_3d_conv=True)
        #self.conv2_3d_encode = Conv(self.num_enc_features, num_3d_features, 3, 1, is_3d_conv=True)
        #self.deconv3d_32_32 = nn.ConvTranspose3d(64,64,(4,4,4),stride = (2,2,2),padding=(1,1,1))
        #
        self.deconv3d_32_1 = Conv(64, 64, 3, 1, is_3d_conv=True)
        #self.deconv3d_32_2 = Conv(64, 64, 3, 1, is_3d_conv=True)
        #self.deconv3d_32_3 = Conv(64, 64, 3, 1, is_3d_conv=True)
        self.deconv3d_32_4 = Conv(64, 64, 3, 1, is_3d_conv=True)

        #self.deconv3d_32_64 = nn.ConvTranspose3d(64,64,(4,4,4),stride = (2,2,2),padding=(1,1,1))
        #self.conv1_3d_decode = Conv(num_3d_features, self.num_dec_features, 3, 1, is_3d_conv=True)
        #self.conv2_3d_decode = Conv(self.num_dec_features, num_3d_features, 3, 1, is_3d_conv=True)
        self.deconv3d_1 = Conv(64, 64, 3, 1, is_3d_conv=True)
        self.deconv3d_2 = Conv(64, 64, 3, 1, is_3d_conv=True)

    def forward(self, num_inputs_to_use, data):
        ###test
        return NULL
        
    def encode(self, x):
        #src_rgb = data['src_rgb_image'][input_idx]
        #src_seg = data['src_seg_image'][input_idx]

        #src_azim_transform_mode = data['src_azim_transform_mode'][input_idx]
        #src_elev_transform_mode = data['src_elev_transform_mode'][input_idx]

        #tgt_azim_transform_mode = data['tgt_azim_transform_mode'][0]
        #tgt_elev_transform_mode = data['tgt_elev_transform_mode'][0]

        #crnt_transform_mode = src_azim_transform_mode - tgt_azim_transform_mode

        #x = src_rgb
        #print("x input shape:", x.shape)

        #if 0 < self.args.num_input_convs:
        x = self.in_conv(x)
        #print("self.in_conv(x):", self.in_conv(x))
        #print("x.shape:", x.shape)
        x1 = self.conv1_2d_encode(x)
        #print("x1.shape:", x1.shape)

        x2 = self.conv2_2d_encode(x1)
        #print("x2.shape:", x2.shape)

        x3 = self.conv3_2d_encode(x2)
        #print("x3.shape:", x3.shape)

        x4 = self.conv4_2d_encode(x3)
        #print("x4.shape:", x4.shape)
        #x5 = self.conv5_2d_encode(x4)
        #print("x5.shape:", x5.shape)
        #skip1 = x1
        #skip2 = x2
        #skip3 = x3
        #skip4 = x4

        #x_out = self.upconv6_2d_encode(x4)
        #x_out = torch.cat((x_out, skip3), 1)
        #x_out = self.iconv6_2d_encode(x_out)
        #print("x_out.shape:", x_out.shape)

        #x_out = self.upconv5_2d_encode(x_out)
        #x_out = torch.cat((x_out, skip3), 1)
        #x_out = self.iconv5_2d_encode(x_out)
        #print("x_out.shape:", x_out.shape)

        x_out = self.upconv4_2d_encode(x4)
        #print("x_out,skip3:", x_out.shape,skip3.shape)
        del x4
        x_out = torch.cat((x_out, x3), 1)
        del x3
        x_out = self.iconv4_2d_encode(x_out)
        #print("x_out.shape:", x_out.shape)

        x_out = self.upconv3_2d_encode(x_out)
        x_out = torch.cat((x_out, x2), 1)

        del x2
        x_out = self.iconv3_2d_encode(x_out)
        #print("x_out.shape:", x_out.shape)

        x_out = self.upconv2_2d_encode(x_out)
        #print("x_out.shape:", x_out.shape)
        x_out = torch.cat((x_out, x1), 1)

        del x1
        x_out = self.iconv2_2d_encode(x_out)
        #print("x_out.shape:", x_out.shape)

        #x_out = self.upconv1_2d_encode(x_out)

        #print("x_out.shape:", x_out.shape)
        #x_out = self.upconv0_2d_encode(x_out)
        #print("x_out.shape:", x_out.shape)

        #if self.src_seg2d is not None:
        #    src_seg2d_output = self.src_seg2d(x_out)
        #    upsample_src_seg2d_output = self.seg_inv_transform(src_seg2d_output)
        #else:
        #upsample_src_seg2d_output = None
        depth = self.vol_dim
        height = x_out.shape[2]
        width = x_out.shape[3]
        #print("output shape size", x_out.shape[0],x_out.shape[1], self.vol_dim)
        x_out = x_out.view(x_out.shape[0],
                           int(x_out.shape[1] / self.vol_dim), self.vol_dim,
                           x_out.shape[2], x_out.shape[3])
        x_out = self.deconv3d_32_1(x_out)
        #x_out = self.deconv3d_32_2(x_out)
        #x_out = self.deconv3d_32_3(x_out)

        x_out = self.deconv3d_32_4(x_out)

        #return x_out

        return x_out
  
    def encode3d(self, x):
       
        #x = src_rgb
        x = self.deconv3d_1(x)
        x = self.deconv3d_2(x)

        return x