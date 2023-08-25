import torch
import torch.nn as nn
import torch.nn.functional as F
from slot_attention import SlotAttention
class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class NeRF(nn.Module):
    def __init__(self,dir_T_size=78,input_feature = 32):
        super(NeRF, self).__init__()
        # linear layer (784 -> 1 hidden node)
        #self.input_size = input_size
       
        #self.fc_t = nn.Linear(dir_T_size, 128)
        self.fc_f = nn.Linear(input_feature, 78)
        #self.fc_h = nn.Linear(global_size, 128)

        
        #self.fc_g = nn.Linear(global_size, 128)
        self.slot_attn = SlotAttention(
        num_slots = 1,
        dim = 256,
        iters = 3   # iterations of attention, defaults to 3
        )
        
        self.fc_1 = nn.Linear(78+dir_T_size,256)
        self.fc_2 = nn.Linear(256,256)
        self.fc_3 = nn.Linear(256,256)
        #self.fc_4 = nn.Linear(256,256)


        self.fc_5 = nn.Linear(256+dir_T_size,256)
        self.fc_6 = nn.Linear(256,256)
        self.fc_7 = nn.Linear(256,256)
        
        self.fc_8 = nn.Linear(512+dir_T_size,256)
        self.fc_9 = nn.Linear(256,256)
        self.fc_10 = nn.Linear(256,3)



        #self.fc_9 = nn.Linear(256+256+256,256)
        #self.fc_10 = nn.Linear(256,256)
        #self.fc_11 = nn.Linear(256,256)
        #self.fc_12 = nn.Linear(256,4)
        #self.fc_c= nn.Linear(color_size, 256)
        #self.fc_9 = nn.Linear(256+256, 256)
        #self.fc_10 = nn.Linear(256,3)
       
        #self.fc_s= nn.Linear(sigma_size, 256)
        #self.fc_11 = nn.Linear(256+256,256)
        #self.fc_12 = nn.Linear(256,1)

    def forward(self, dir_T,input_feature,n_views):

        #t = self.fc_t(dir_T)
        #print("input_feature:", input_feature.shape)
        #t = self.fc_t(dir_T)
        #t = t.repeat(3,1)

        x = dir_T.repeat(n_views,1)
        input_feature = F.relu(self.fc_f(input_feature))
        #h = self.fc_h(global_feature)
        #h = F.relu(self.fc_h(global_feature))

        #inputi = self.fc_i(inputi)
        
        #print(global_f.shape, f.shape)
        #x = F.relu(self.fc_xy2(x),inplace=True)
        #x = F.relu(self.fc_xy3(x),inplace=True)
        #x = F.relu(self.fc_xy4(x),inplace=True)
        x = torch.cat([input_feature,x],dim = -1)

        #f = torch.cat([dir_T,f],dim = -1)
        #f = torch.cat([dir_T,f],dim = -1)

        #f = torch.cat([h,f],dim = -1)
        del input_feature
        #print(f.shape)
        x = F.relu(self.fc_1(x),inplace=True)
        
        x = F.relu(self.fc_2(x),inplace=True)
        x = F.relu(self.fc_3(x),inplace=True)
        #x = F.relu(self.fc_4(x),inplace=True)

        #x = torch.cat([f,x],dim = -1)
        #print("x:", x.shape)
        x = x.reshape(n_views,-1,x.shape[-1])
        x = x.permute(1,0,2)
        x = self.slot_attn(x)
        #x = torch.mean(x, dim = 0)
        #print("x:", x.shape)
        x = x.reshape(-1,x.shape[-1])
        
        x = torch.cat([x,dir_T],dim = -1)
        del dir_T
        y = F.relu(self.fc_5(x),inplace=True)
        y = F.relu(self.fc_6(y),inplace=True)
        y = F.relu(self.fc_7(y),inplace=True)


        y = torch.cat([x,y],dim = -1)
        del x 
        y = F.relu(self.fc_8(y),inplace=True)
        y = F.relu(self.fc_9(y),inplace=True)
        m = nn.Sigmoid()
        y = m(self.fc_10(y))
        

        #rgb = y[:,:3]
        #sigma = y[:,3:]

        #del y
        return y



