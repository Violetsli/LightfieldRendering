# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#from tdutils import *

def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
        
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std

def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))

def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]

def initseq(s):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])

class Rodrigues(nn.Module):
    def __init__(self):
        super(Rodrigues, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), dim=1).view(-1, 3, 3)

class Quaternion(nn.Module):
    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        return torch.stack((
            1. - 2. * rvec[:, 1] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 1] ** 2
            ), dim=1).view(-1, 3, 3)

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        ninputs = 1
        tied = False
        self.ninputs = ninputs
        #print("ninputs.shape",ninputs.shape)
        self.tied = tied

        self.down1 = nn.ModuleList([nn.Sequential(
                nn.Conv2d(3, 16, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                #nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2),
                #nn.Conv2d(512, 1024, 4, 2, 1), nn.LeakyReLU(0.2)
                ) ])
                #for i in range(1 if self.tied else self.ninputs)])
        self.down2 = nn.Sequential(
                nn.Linear(256, 512), nn.LeakyReLU(0.2))
        height, width = 128, 128
        ypad = ((height + 127) // 128) * 128 - height
        xpad = ((width + 127) // 128) * 128 - width
        self.pad = nn.ZeroPad2d((xpad // 2, xpad - xpad // 2, ypad // 2, ypad - ypad // 2))
        self.mu = nn.Linear(512, 256)
        self.logstd = nn.Linear(512, 256)

        #for i in range(1 if self.tied else self.ninputs):
        #    tdutils.initseq(self.down1[i])
        initseq(self.down1[0])
        initseq(self.down2)
        initmod(self.mu)
        initmod(self.logstd)

    def forward(self, x):
        #print ("x.shape",x.shape)
        x = self.pad(x)
       # x = [self.down1[0 if self.tied else i](x[:, i*3:(i+1)*3, :, :]).view(-1, 256 * 3 * 4) for i in range(self.ninputs)]
        #x = [self.down1[0](x[:, 0:3, :, :]).view(-1, 256)]
        x = [self.down1[0](x[:, 0:3, :, :])]
        x = torch.cat(x, dim=1)
        #print("output_down1.shape",x.shape)
        #x = self.down2(x)
        #print("down2.shape",x.shape)
        #mu, logstd = self.mu(x) * 0.1, self.logstd(x) * 0.01
        #if self.training:
        #    z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        #else:
        #    z = mu
        #z = self.mu(x) * 0.1
        #losses = {}
        #if "kldiv" in losslist:
        #    losses["kldiv"] = torch.mean(-0.5 - logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)
        print("x.shape:", x.shape)
        return {"encoding": x}

class ConvTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=68, templateres=128):
        super(ConvTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        # build template convolution stack
        self.template1 = nn.Sequential(nn.Linear(self.encodingsize, 256), nn.LeakyReLU(0.2))
        template2 = []
        inchannels, outchannels = 1024,256
        #print("layers:",int(np.log2(self.templateres)) - 1)
        for i in range(int(np.log2(self.templateres)) - 1):
            template2.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
            template2.append(nn.LeakyReLU(0.2))
            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels
        template2.append(nn.ConvTranspose3d(inchannels, 68, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)

        for m in [self.template1, self.template2]:
            initseq(m)

    def forward(self, encoding):
        #print("encoding:",encoding.shape)
        #print("encoding.view(-1, 1024, 1, 1, 1):", encoding.view(-1, 1024, 1, 1, 1).shape)
        return self.template2(encoding.view(-1, 1024, 1, 1, 1))

class LinearTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=4, templateres=128):
        super(LinearTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        self.template1 = nn.Sequential(
            nn.Linear(self.encodingsize, 8), nn.LeakyReLU(0.2),
            nn.Linear(8, self.templateres ** 3 * self.outchannels))

        for m in [self.template1]:
            initseq(m)

    def forward(self, encoding):
        return self.template1(encoding).view(-1, self.outchannels, self.templateres, self.templateres, self.templateres)

def gettemplate(templatetype, **kwargs):
    if templatetype == "conv":
        return ConvTemplate(**kwargs)
    elif templatetype == "affinemix":
        return LinearTemplate(**kwargs)
    else:
        return None

class ConvWarp(nn.Module):
    def __init__(self, displacementwarp=False, **kwargs):
        super(ConvWarp, self).__init__()

        self.displacementwarp = displacementwarp

        self.warp1 = nn.Sequential(
                nn.Linear(256, 1024), nn.LeakyReLU(0.2))
        self.warp2 = nn.Sequential(
                nn.ConvTranspose3d(1024, 512, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(512, 512, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(512, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(256, 3, 4, 2, 1))
        for m in [self.warp1, self.warp2]:
            initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(
                np.linspace(-1.0, 1.0, 32),
                np.linspace(-1.0, 1.0, 32),
                np.linspace(-1.0, 1.0, 32), indexing='ij')
        self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=0)[None].astype(np.float32)))

    def forward(self, encoding):
        finalwarp = self.warp2(self.warp1(encoding).view(-1, 1024, 1, 1, 1)) * (2. / 1024)
        if not self.displacementwarp:
            finalwarp = finalwarp + self.grid
        return finalwarp

class AffineMixWarp(nn.Module):
    def __init__(self, **kwargs):
        super(AffineMixWarp, self).__init__()

        self.quat = Quaternion()

        self.warps = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 3*16))
        self.warpr = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 4*16))
        self.warpt = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 3*16))
        self.weightbranch = nn.Sequential(
                nn.Linear(256, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 16*32*32*32))
        for m in [self.warps, self.warpr, self.warpt, self.weightbranch]:
            initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(
                np.linspace(-1.0, 1.0, 32),
                np.linspace(-1.0, 1.0, 32),
                np.linspace(-1.0, 1.0, 32), indexing='ij')
        self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

    def forward(self, encoding):
        warps = self.warps(encoding).view(encoding.size(0), 16, 3)
        warpr = self.warpr(encoding).view(encoding.size(0), 16, 4)
        warpt = self.warpt(encoding).view(encoding.size(0), 16, 3) * 0.1
        warprot = self.quat(warpr.view(-1, 4)).view(encoding.size(0), 16, 3, 3)

        weight = torch.exp(self.weightbranch(encoding).view(encoding.size(0), 16, 32, 32, 32))

        warpedweight = torch.cat([
            F.grid_sample(weight[:, i:i+1, :, :, :],
                torch.sum(((self.grid - warpt[:, None, None, None, i, :])[:, :, :, :, None, :] *
                    warprot[:, None, None, None, i, :, :]), dim=5) *
                    warps[:, None, None, None, i, :], padding_mode='border')
            for i in range(weight.size(1))], dim=1)

        warp = torch.sum(torch.stack([
            warpedweight[:, i, :, :, :, None] *
            (torch.sum(((self.grid - warpt[:, None, None, None, i, :])[:, :, :, :, None, :] *
                warprot[:, None, None, None, i, :, :]), dim=5) *
                warps[:, None, None, None, i, :])
            for i in range(weight.size(1))], dim=1), dim=1) / torch.sum(warpedweight, dim=1).clamp(min=0.001)[:, :, :, :, None]

        return warp.permute(0, 4, 1, 2, 3)

def getwarp(warptype, **kwargs):
    if warptype == "conv":
        return ConvWarp(**kwargs)
    elif warptype == "affinemix":
        return AffineMixWarp(**kwargs)
    else:
        return None

class Decoder(nn.Module):
    def __init__(self, templatetype="conv", templateres=128,
            viewconditioned=False, globalwarp=True, warptype="affinemix",
            displacementwarp=False):
        super(Decoder, self).__init__()

        self.templatetype = templatetype
        self.templateres = templateres
        self.viewconditioned = viewconditioned
        self.globalwarp = globalwarp
        self.warptype = warptype
        self.displacementwarp = displacementwarp

        if self.viewconditioned:
            self.template = gettemplate(self.templatetype, encodingsize=256+3,
                    outchannels=3, templateres=self.templateres)
            self.templatealpha = gettemplate(self.templatetype, encodingsize=256,
                    outchannels=1, templateres=self.templateres)
        else:
            self.template = gettemplate(self.templatetype, templateres=self.templateres)

        self.warp = getwarp(self.warptype, displacementwarp=self.displacementwarp)

        if self.globalwarp:
            self.quat = Quaternion()

            self.gwarps = nn.Sequential(
                    nn.Linear(256, 128), nn.LeakyReLU(0.2),
                    nn.Linear(128, 3))
            self.gwarpr = nn.Sequential(
                    nn.Linear(256, 128), nn.LeakyReLU(0.2),
                    nn.Linear(128, 4))
            self.gwarpt = nn.Sequential(
                    nn.Linear(256, 128), nn.LeakyReLU(0.2),
                    nn.Linear(128, 3))

            #initseq = initseq
            for m in [self.gwarps, self.gwarpr, self.gwarpt]:
                initseq(m)

    def forward(self, encoding):
        #scale = torch.tensor([25., 25., 25., 1.], device=encoding.device)[None, :, None, None, None]
        #bias = torch.tensor([100., 100., 100., 0.], device=encoding.device)[None, :, None, None, None]
        #viewpos = viewpos.permute(1,0)
        #print("viewpos.shape", viewpos.shape)
        # run template branch
        #viewdir = viewpos / torch.sqrt(torch.sum(viewpos ** 2, dim=-1, keepdim=True))
        #templatein = torch.cat([encoding, viewdir], dim=1) if self.viewconditioned else encoding
        
        templatein = encoding
        #print("encoding.shape", encoding.shape)
        template = self.template(templatein)
        #print("template.shape",template.shape)
        #viewconditioned == false
        if self.viewconditioned:
            print("self.viewconditioned")
            # run alpha branch without viewpoint information
            template = torch.cat([template, self.templatealpha(encoding)], dim=1)
        # scale up to 0-255 range approximately
        #template = F.softplus(bias + scale * template)
        #template = F.softplus(template)
        # compute warp voxel field
        #warp = self.warp(encoding) if self.warp is not None else None
        #print("template,",template.shape)
        #if self.globalwarp:
            # compute single affine transformation
        #    gwarps = 1.0 * torch.exp(0.05 * self.gwarps(encoding).view(encoding.size(0), 3))
        #    gwarpr = self.gwarpr(encoding).view(encoding.size(0), 4) * 0.1
        #    gwarpt = self.gwarpt(encoding).view(encoding.size(0), 3) * 0.025
        #    gwarprot = self.quat(gwarpr.view(-1, 4)).view(encoding.size(0), 3, 3)

       # losses = {}

        # tv-L1 prior
        #if "tvl1" in losslist:
        #    logalpha = torch.log(1e-5 + template[:, -1, :, :, :])
        #    losses["tvl1"] = torch.mean(torch.sqrt(1e-5 +
        #        (logalpha[:, :-1, :-1, 1:] - logalpha[:, :-1, :-1, :-1]) ** 2 +
        #        (logalpha[:, :-1, 1:, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2 +
        #        (logalpha[:, 1:, :-1, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2))

        #return {"template": template, "warp": warp,
        #        **({"gwarps": gwarps, "gwarprot": gwarprot, "gwarpt": gwarpt} if self.globalwarp else {}),
        #        "losses": losses}
        return {"template": template}



#----------------my way to use it---------
"""
        td_encoder = Encoder()
        td_encoder.to(device=device)
        encoderd = td_encoder(x)
        encoded_latent = encoderd["encoding"]
      
        #print(encoded_latent.shape)
       
        td_decoder = Decoder()
        td_decoder.to(device=device)
        #print("td_decoder():",Decoder())
        decoded = td_decoder(encoded_latent)

        decoded_latent = decoded["template"]  """