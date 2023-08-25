import torch
import torch.nn as nn
import torch.nn.functional as F
#import models.tbn432_small as tbnlib
from numpy import *
import os
#import models.nerf_add as mlplib
#import models.enr as enrlib
from models.enr_64 import ResNet2d, ResNet3d,InverseProjection,Projection
import models.nerf_addcolor_nosig as mlplib
#import models.nerf_addcolor_nosig_mulm_order as mlplib
#from models.egiraffe314 import NeuralRenderer
import numpy as np
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        #Mlp = mlplib.NeRF(6,32,3)   #142
        #self.Mlp = Mlp
        # igrida 64 points 64cube has mlp,not using mlp
        img_shape = [3, 128, 128]
        a_shape = [12, 128, 128]
        channels_2d = [52]
        strides_2d = [1]
        channels_2d_a = [12]
        strides_2d_a = [1]
        channels_2d_1 = [64,64,128, 128, 128, 128, 256, 256, 128, 128, 128]
        strides_2d_1 = [1,1,2,1, 2, 1, 2,1, -2, 1, 1]
        #channels_2d = [64, 64, 128, 128, 128, 128]
        #strides_2d = [1, 1, 2, 1, 2, 1]
        channels_3d = [32, 32, 64, 64, 64, 32, 32, 32]
        strides_3d = [1, 1, 2, 1, 1, -2, 1, 1]
        #channels_3d = [32, 32, 128, 128, 128, 64, 64, 64]
        #strides_3d = [1, 1, 2, 1, 1, -2, 1, 1]
        num_channels_projection = [512, 256, 256]
        num_channels_inv_projection = [256, 512, 2048]
        self.inv_transform_2d_a = ResNet2d(a_shape, channels_2d_a,
                                         strides_2d_a)
        self.inv_transform_2d = ResNet2d(img_shape, channels_2d,
                                         strides_2d)
        #input_shape_1 = self.inv_transform_2d.output_shape
        img_shape_1 = [64, 128, 128]
        self.inv_transform_2d_1 = ResNet2d(img_shape_1, channels_2d_1,
                                         strides_2d_1)
        input_shape = self.inv_transform_2d_1.output_shape

        self.inv_projection = InverseProjection(input_shape, num_channels_inv_projection)
        # Transform 3D inverse projection into a scene representation
        self.inv_transform_3d = ResNet3d((35,32,32,32),
                                         channels_3d, strides_3d)
        
        Embedding_pk = mlplib.Embedding(2,6)
        #self.tbn = tbn
        #self.fc_globalf1 = nn.Linear(3072, 1024)
        #self.fc_globalf2 = nn.Linear(1024, 256)
        #self.Embedding_pk = Embedding_pk
        self.Embedding_pk = Embedding_pk
        Mlp = mlplib.NeRF(78,30,30)   #142
        self.Mlp = Mlp
        #self.nrendering = NeuralRenderer()
        self.conv_up = nn.Conv2d(31, 3, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.upsampling = nn.Upsample(scale_factor = 4,mode='bilinear', align_corners = True)
    def write_obj(self, mesh_v, filepath,r,g,b):
        with open(filepath, 'a') as fp:
            for i in range(mesh_v.shape[0]):

                s = 'v {} {} {} {} {} {}\n'.format(mesh_v[i, 0], mesh_v[i, 1], mesh_v[i, 2], r, g, b)
                fp.write(s)

    def sample_fine(self, n_views, camtrans_tar, camdir_tar_, 
        image_src, camrot_src, camtrans_src,weights,all_zs,coarse_number,fine_number,z_samp_depth,z_far,z_near):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        #device = rays.device
        B,NV,_,H,W = image_src.shape
        weights = weights.detach() + 1e-5  # Prevent disample_finevision by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        #print("weights:", weights.shape)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (B, Kc+1)
        #print("cdf",cdf.shape)
        u = torch.rand(
            cdf.shape[0],cdf.shape[1],cdf.shape[2],cdf.shape[3], fine_number)  # (B, Kf)
        #print("u",u.shape)
        inds = torch.searchsorted(cdf.to(image_src.device), u.to(image_src.device), right=True).float() - 1.0  # (B, Kf)
        #print("inds",inds.shape)
        inds = torch.clamp_min(inds, 0.0)
        #print("inds",inds.shape)
        z_steps = (inds + torch.rand_like(inds)) / coarse_number  # (B, Kf)  32 coarse
        #print("inds",z_steps.shape)
        
        #, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        #if not self.lindisp:  # Use linear sampling in depth space
        #    z_samp = z_near * (1 - z_steps) + z_far * z_steps  # (B, Kf)
        #else:  # Use linear sampling in disparity space
        #z_samp = 1 / (1 / z_near * (1 - z_steps) + 1 / z_far * z_steps) 
        #if not self.lindisp:  # Use linear sampling in depth space
        z_samp = z_near * (1 - z_steps) + z_far * z_steps  # (B, Kf)
        #else:  # Use linear sampling in disparity space
        #    z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        z_samp_depth = z_samp_depth.unsqueeze(-1).repeat((1,1,1,1,int(fine_number/3)))
        z_samp_depth = z_samp_depth + torch.randn_like(z_samp_depth) * 0.01
        #z_samp_depth = z_samp_depth + torch.randn_like(z_samp_depth) * 0.01

        # Clamp does not support tensor bounds
        #print(z_samp_depth)
        #print(z_samp.type(),z_samp_depth.type())
        #z_samp, argsort = torch.sort(z_samp, dim=-1)
        z_samp_depth = torch.max(torch.min(z_samp_depth, z_far*torch.ones_like(z_samp_depth)), z_near*torch.ones_like(z_samp_depth))
        #print(z_far,z_near,z_samp_depth )
        #all_zs = all_zs.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        #all_zs = all_zs.repeat(1,z_samp.shape[1],z_samp.shape[2],z_samp.shape[3],1)
        #print("z_samp:", all_zs.shape, z_samp.shape)
        #z_samp = torch.cat([all_zs, z_samp], -1)  # (B, K)
        z_samp = torch.cat([z_samp_depth, z_samp], -1)  # (B, K)
        del z_samp_depth
        z_samp, argsort = torch.sort(z_samp, dim=-1)
        #print("z_samp:", z_samp.shape)
        
        #print("z_samp:", z_samp.shape)
        #print(camtrans_tar.shape,z_samp.shape, camdir_tar_.shape)
        #torch.Size([1, 3]) torch.Size([1, 1, 150, 200, 64]) torch.Size([1, 150, 200, 3])
        point_tar = camtrans_tar[:, None,None,None,None,:] +  z_samp[:,:,:,:,:,None]*camdir_tar_[:,None,:,:,None,:]
        point_tar = point_tar.permute(0,1,4,2,3,5)
        #point_tar.detach()
        #point_tar.requires_grad = False
        #print("z_samp:",point_tar.shape,camtrans_src.shape)
        point_tar = point_tar - camtrans_src[:,:,None,None,None,:]   
        #camrot_src = torch.transpose(camrot_src,3,2)
        #print("shape:", camrot_src.shape,point_tar.shape)
        point_tar = torch.matmul(camrot_src[:,:,None,None,None,:,:], point_tar.unsqueeze(-1).to(torch.float32))[..., 0]     
        #point_tar[...,0] =  -2*point_tar[...,0]*z_far/ (2.6408*point_tar[...,2])
        #point_tar[...,1] =  2*point_tar[...,1]*z_far/ (2.0112*point_tar[...,2])
        point_tar[...,0] =  -2*point_tar[...,0]*z_far/ (2.9063*point_tar[...,2])
        point_tar[...,1] =  2*point_tar[...,1]*z_far/ (2.22885*point_tar[...,2])
        point_tar[...,2] =  -2*point_tar[...,2]/(z_far-z_near) - (z_far+z_near)/(z_far-z_near)
        #print("point_tar:",point_tar.shape)
        #print(z_samp)
        return point_tar,z_samp


    def forward(self, n_views, camrot_tar, camtrans_tar, camdir_tar, intrinsic_tar, 
        image_src, camrot_src, camtrans_src,camdir_src, z_near,z_far):
        B,NV,_,H,W = image_src.shape
        z_step = 0
        coarse_number = 64   # coarsenumber
        fine_number = 128  #128
        #depth_number = 32

        all_zs = torch.tensor(linspace(z_near,z_far,coarse_number))
        all_zs = all_zs.unsqueeze(0).to(camrot_tar.device)
        #print(all_zs.shape)
        deltas_z = (z_far-z_near)/coarse_number
        #all_zs[:,:Width-1] = all_zs[:,:Width-1]+random.randint(1,100)*deltas_z/100
        #print(all_zs)
       
        all_zs[...,:coarse_number-1] = all_zs[...,:coarse_number-1]+random.randint(1,100)*deltas_z/100
        
        deltas_z = all_zs[..., 1:] - all_zs[..., :-1]
        #deltas_z = torch.cat([deltas_z, torch.Tensor([1e10]).to(camrot_tar.device).expand(deltas_z[..., :1].shape)], -1)  # [N_rays, N_samples]
        delta_inf = z_far - all_zs[..., -1:]
        deltas_z = torch.cat([deltas_z, delta_inf.to(camrot_tar.device)], -1)  # [N_rays, N_samples]
        del delta_inf
        #delta_inf = torch.Tensor([z_far]).to(camrot_tar.device).expand(all_zs[:,-1:].shape) - all_zs[:, -1:]
        #deltas_z = torch.cat([deltas_z, delta_inf], -1)  # [N_rays, N_samples]
        deltas_z = deltas_z.float()
        
        camdir_tar_ = camdir_tar.permute(0,3,1,2)
        #print(camdir_tar.shape)
        camdir_tar_ = F.interpolate(camdir_tar_, scale_factor=0.25, mode="bilinear",align_corners = True)
        camdir_tar_ = camdir_tar_.permute(0,2,3,1)

        #camdir_tar_ = camdir_tar
        #---------------------------------------------------------------
        #-----------------------------------------------------------------------
        point_tar = camtrans_tar[:, None,None,None,:] +  all_zs[:,:,None,None,None].to(camtrans_tar.device)*camdir_tar_[:,None,:,:,:]
        point_tar.requires_grad = False
        point_tar = point_tar[:,None,:,:,:,:] - camtrans_src[:,:,None,None,None,:]   
        
        camrot_src = torch.transpose(camrot_src,3,2)
        point_tar = torch.matmul(camrot_src[:,:, None,None,None,:,:], point_tar.unsqueeze(-1).to(torch.float32))[..., 0]     
        #----------------------------------------------------------------------
        #camrot_tar_temp = [t.inverse() for t in torch.functional.split(camrot_tar,1)]
        #camrot_tar_temp = torch.stack(camrot_tar_temp,dim=0).squeeze(1)
        #C_tar = -torch.matmul(camrot_tar_temp, camtrans_tar.unsqueeze(-1))[..., 0] 
        #del camrot_tar_temp,camtrans_tar
        #C_tar = C_tar.unsqueeze(1)
        #C_tar = C_tar.unsqueeze(1)
        #print(C.shape)
        T_tar = camtrans_tar.unsqueeze(1).unsqueeze(1)
        T_tar = T_tar.repeat(1,H,W,1).to(camrot_tar.device) 
        #print(C.shape)
        C_d_tar = torch.cross(T_tar,camdir_tar,dim=-1)
        PK_tar = torch.cat([camdir_tar,C_d_tar],dim=-1)
        del T_tar,C_d_tar
        #-------------------------------------------------
        """
        T_tar_src = T_tar[:,None,:,:,:]- camtrans_src[:,:,None,None,:]  
        #D_tar = camdir_tar.unsqueeze(1).unsqueeze(-1)
        #print(camrot_src.shape,D_tar.shape )
        D_tar  = torch.matmul(camrot_src[:,:,None,None,:,:],camdir_tar[:,None,:,:,:,None])[..., 0]  

        C_d_tar_src =  torch.cross(T_tar_src,D_tar,dim=-1)
        PK_tar_src = torch.cat([D_tar,C_d_tar_src],dim=-1)

        PK_tar_src = PK_tar_src.reshape(B*n_views,6,H,W)

        #PK_tar_src = PK_tar[:,None,:,:,:] - camtrans_src[:,:,None,None,:]   
        #PK_tar_src = torch.matmul(camrot_src,PK_tar_src.unsqueeze(-1))[..., 0]    
        del T_tar,C_d_tar,D_tar,T_tar_src
        """
        #---------------------------------------------------------------------
        #----------------------------------------------------------------------
        R_ti = camrot_src * camrot_tar[:,None,:,:]
        T_ti =camtrans_tar[:,None,:] - camtrans_src
        T_ti = torch.matmul(camrot_src,T_ti.unsqueeze(-1))[..., 0]    
        R_ti = R_ti.reshape(B,n_views,9)
        R_ti = R_ti.reshape(B*n_views,9,1,1)
        T_ti = T_ti.reshape(B*n_views,3,1,1)
        RT_ti =  torch.cat((R_ti, T_ti), 1)
        del R_ti,T_ti
        
        #----------------------------------------------------------------------
        #point_tar[...,0] =  -2*point_tar[...,0]*z_far/ (2.6408*point_tar[...,2])
        #point_tar[...,1] =  2*point_tar[...,1]*z_far/ (2.0112*point_tar[...,2])
        point_tar[...,0] =  -2*point_tar[...,0]*z_far/ (2.9063*point_tar[...,2])
        point_tar[...,1] =  2*point_tar[...,1]*z_far/ (2.22885*point_tar[...,2])
        point_tar[...,2] =  -2*point_tar[...,2]/(z_far-z_near) - (z_far+z_near)/(z_far-z_near)

        image_src_ = image_src.reshape(B*n_views,3,H,W)
        #decout_1 = self.inv_transform_2d(image_src_)

        image_src_ = torch.cat((image_src_[:,:,0,:].unsqueeze(-2).repeat(1,1,4,1),image_src_), 2)

        #supp = torch.zeros([image_src_.shape[0], image_src_.shape[1],2,W])
        #image_src_ = torch.cat((supp.to(image_src.device),image_src_), 2)
        #image_src_ = torch.cat((image_src_, supp.to(image_src.device)), 2)
        #-------------------------------------------------------------------------------------------------
        decout = self.inv_transform_2d(image_src_)
        
        
        #RT_ti = self.inv_transform_2d_a(RT_ti.repeat(1,1,decout.shape[2],decout.shape[3]))
        decout  =  torch.cat((decout,RT_ti.repeat(1,1,decout.shape[2],decout.shape[3])), 1)
        #decout = self.inv_transform_2d_1(decout)
        del RT_ti
        #decout_1 = decout
        #------------------------------
        #supp1 = torch.zeros([PK_tar_src.shape[0], PK_tar_src.shape[1],2,W]).to(image_src.device)

        #PK_tar_src = torch.cat((supp1,PK_tar_src), 2)
        #PK_tar_src = torch.cat((PK_tar_src,supp1), 2)

        #print("decout.shape",decout.shape,RT_ti.shape )
        #decout  =  torch.cat((decout,PK_tar_src), 1)

        #----------------------------------------------
        decout = self.inv_transform_2d_1(decout)
        #del supp,image_src_
        decout = self.inv_projection(decout)
        #print("decout23",decout.shape)

        #image_src_ = image_src.reshape(B*n_views,3,H,W)
        image_src_ = F.interpolate(image_src.reshape(B*n_views,3,H,W),(int(H/4),int(W/4)), mode='bilinear', align_corners=True) #.expand(1,1,features.shape[2],1,1)
        image_src_ = torch.cat((image_src_[:,:,0,:].unsqueeze(-2),image_src_),2)
        image_src_ = image_src_.unsqueeze(2).expand(-1,-1,decout.shape[2],-1,-1)
        decout = torch.cat((image_src_.reshape(B,n_views,-1, decout.shape[-3], decout.shape[-2],decout.shape[-1]),decout.reshape(B,n_views,decout.shape[1], decout.shape[2],decout.shape[3],decout.shape[4])),2)
        del image_src_
        decout = self.inv_transform_3d(decout.reshape(B*n_views,decout.shape[2], decout.shape[3],decout.shape[4],decout.shape[5]))
        Cube_channel = decout.shape[1]
        
        #decout = decout[:,:,:,1:,:]   # N,C,D,H,W 

        sample_all = F.grid_sample(decout[:,:Cube_channel-1,:,1:,:], point_tar.reshape(B*n_views,coarse_number,point_tar.shape[-3],point_tar.shape[-2],3), padding_mode = 'border',align_corners=True)
        #print(sample_all.shape)   #torch.Size([8, 32, 128, 128, 64])
        sample_all_c = F.grid_sample(decout[:,Cube_channel-1:,:,1:,:], point_tar.reshape(B*n_views,coarse_number,point_tar.shape[-3],point_tar.shape[-2],3), padding_mode = 'zeros',align_corners=True)

        sample_all = sample_all.reshape(B,n_views,Cube_channel-1,coarse_number,sample_all.shape[-2],sample_all.shape[-1])
        sample_all_c = sample_all_c.reshape(B,n_views,1,coarse_number,sample_all.shape[-2],sample_all.shape[-1])         
        m_soft = nn.Softmax(dim=1)
        sample_all_c = m_soft(sample_all_c)
        sample_all = sample_all.mul(sample_all_c)
        sample_all = torch.sum(sample_all, dim = 1)
        #hidden = hidden.reshape(B, n_views,-1)
        #hidden,_ = torch.max(hidden, dim = 1)
        Cube_channel = sample_all.shape[1]
        del sample_all_c
        sample_all = sample_all.permute(0,1,3,4,2)
        #------------------------------------------------------------------------------------------------------
        
        deltas_z = deltas_z * torch.norm(camdir_tar_.unsqueeze(-2), dim=-1)
        #print("deltas_z:", deltas_z.shape)
        alphas = 1-torch.exp(-deltas_z.unsqueeze(1)*torch.relu(sample_all[:, (Cube_channel-1):, :, :, :])) # (N_rays, N_samples_)
 
        alphas_shifted = torch.cat([torch.ones_like(alphas[...,:1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        #print(alphas_shifted.shape)
        weights = alphas * torch.cumprod(alphas_shifted, -1)[..., :-1] # (N_rays, N_samples_)
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        #alphas_final = torch.sum(weights, -1)
        Cube_channel = sample_all.shape[1]
        #deltas_z = deltas_z * torch.norm(camdir_tar_.unsqueeze(-2), dim=-1)
        #fine_point_tar = torch.cat((fine_point_tar, point_tar.to(image_src.device)), 2)
        #fine_point_tar = fine_point_tar.permute(0,1,3,4,2,5)
        #deltas_z = fine_point_tar[:,:,:,:,1:,2] - fine_point_tar[:,:,:,:,:-1,2]
        #del fine_point_tar,point_tar
        alphas_final= torch.sum(weights, -1)
       
        
        depth_final_coarse = torch.sum(weights * all_zs[:,None,None,None, :], -1) #/(alphas_final+1e-8) #/alphas_final  # (B, Kc)  # (B)
        

        feature_final_coarse = torch.sum(weights*sample_all[:, :(Cube_channel-1), :, :, :], -1) # (N_rays, 3)
        #print("feature_final:",feature_final.type())
        feature_final_coarse = torch.cat([feature_final_coarse,alphas_final],dim=1)
        #depth_final_coarse = depth_final_coarse.float()
        #print("depth_final_coarse",depth_final_coarse.shape)
        #feature_final_coarse = torch.cat([depth_final_coarse,feature_final_coarse],dim=1)

        del alphas,sample_all

        #feature_final_coarse = m1(feature_final_coarse)
        #conv_rgb_coarse = self.conv_up(feature_final_coarse)
        conv_rgb_coarse = self.upsampling(self.conv_up(feature_final_coarse))

        #  #torch.Size([n_views, 3, 128, 128])
        feature_final_coarse =self.upsampling(feature_final_coarse)
        alphas_final= feature_final_coarse[:,Cube_channel-1:,:,:]

        feature_final_coarse = feature_final_coarse[:,:Cube_channel-1,:,:]
        ray_rgb_coarse = feature_final_coarse[:,:3,:,:]
        feature_final_coarse = feature_final_coarse.permute(0, 2, 3, 1)

        #torch.Size([1, 1, 150, 200, 64]) torch.Size([1, 64])
        #print("alphas_final,depth_final_coarse:",weights.shape, alphas_final.shape)
        del point_tar
        
        #print(depth_final.shape)
        #torch.Size([1, 1, 1, 150, 200])
        #-------------------------------------------------------------------------------------------
        #print("weights:", weights.shape)
        fine_point_tar, z_samp= self.sample_fine(n_views, camtrans_tar, camdir_tar_, 
        image_src, camrot_src, camtrans_src,weights,all_zs,coarse_number,fine_number,depth_final_coarse,z_far,z_near)
        #fine_point_tar.requires_grad = False
        depth_final_coarse = self.upsampling(depth_final_coarse)
        #print(depth_final_coarse.shape)
        depth_final_coarse = torch.max(torch.min(depth_final_coarse, z_far*torch.ones_like(depth_final_coarse)), torch.zeros_like(depth_final_coarse))

        #print("fine_samples:",fine_point_tar.shape, point_tar.shape)
        #----------------------------------------------------------------------------------
        Cube_channel = decout.shape[1]
        #fine_point_tar = torch.cat((fine_point_tar, point_tar.to(image_src.device)), 2)
        #print("fine_samples:",fine_point_tar.shape)
        Width = fine_point_tar.shape[2]
        sample_all_fine = F.grid_sample(decout[:,:Cube_channel-1,:,1:,:], fine_point_tar.reshape(B*n_views,Width,fine_point_tar.shape[-3],fine_point_tar.shape[-2],3), padding_mode = 'border',align_corners=True)
        #print(sample_all.shape)   #torch.Size([8, 32, 128, 128, 64])
        sample_all_c_fine = F.grid_sample(decout[:,Cube_channel-1:,:,1:,:], fine_point_tar.reshape(B*n_views,Width,fine_point_tar.shape[-3],fine_point_tar.shape[-2],3), padding_mode = 'zeros',align_corners=True)
        del decout
        sample_all_fine = sample_all_fine.reshape(B,n_views,Cube_channel-1,Width,sample_all_fine.shape[-2],sample_all_fine.shape[-1])
        sample_all_c_fine = sample_all_c_fine.reshape(B,n_views,1,Width,sample_all_fine.shape[-2],sample_all_fine.shape[-1])
        #print("sample_all_fine:",sample_all_fine.type())

        m_soft = nn.Softmax(dim=1)
        sample_all_c_fine = m_soft(sample_all_c_fine)
        sample_all_fine = sample_all_fine.mul(sample_all_c_fine)
        sample_all_fine = torch.sum(sample_all_fine, dim = 1)
        sample_all_fine = sample_all_fine.permute(0,1,3,4,2)
        #print("sample_all_fine:",sample_all_fine.type())
        del sample_all_c_fine
        #sample_all_fine = torch.cat((sample_all,sample_all_fine), -1)
        #-----------------------------------------------------------------------------------------
        Cube_channel = sample_all_fine.shape[1]
        #deltas_z = deltas_z * torch.norm(camdir_tar_.unsqueeze(-2), dim=-1)
        #fine_point_tar = torch.cat((fine_point_tar, point_tar.to(image_src.device)), 2)
        #fine_point_tar = fine_point_tar.permute(0,1,3,4,2,5)
        #deltas_z = fine_point_tar[:,:,:,:,1:,2] - fine_point_tar[:,:,:,:,:-1,2]
        del  fine_point_tar
        #deltas_z = torch.cat([deltas_z.unsqueeze(1),z_samp], -1)
        #print(z_samp.shape,sample_all_fine.shape)
        deltas_z_fine = z_samp[..., 1:] - z_samp[..., :-1]  # (B, K-1)
        #print("z_samp:",z_samp.type())
        #z_samp = z_samp.float()
        #print(deltas_z_fine)


        #deltas_z_fine = all_zs[:, 1:] - all_zs[:, :-1]
        #deltas_z = torch.cat([deltas_z, torch.Tensor([1e10]).to(camrot_tar.device).expand(deltas_z[..., :1].shape)], -1)  # [N_rays, N_samples]
        delta_inf = z_far - z_samp[..., -1:]
        deltas_z_fine = torch.cat([deltas_z_fine, delta_inf.to(deltas_z_fine.device)], -1)  # [N_rays, N_samples]
        del delta_inf
        #deltas_z_fine = torch.cat([deltas_z_fine, torch.Tensor([1e10]).to(camrot_tar.device).expand(deltas_z_fine[..., :1].shape)], -1)  # [N_rays, N_samples]
        alphas_fine = 1-torch.exp(-deltas_z_fine*torch.relu(sample_all_fine[:, (Cube_channel-1):, :, :, :])) # (N_rays, N_samples_)
        #print("alphas_fine:",alphas_fine.type())
        alphas_fine = alphas_fine.float()
        alphas_shifted_fine = torch.cat([torch.ones_like(alphas_fine[...,:1]), 1-alphas_fine+1e-10], -1) # [1, a1, a2, ...]
        
        #print("alphas_shifted_fine:",alphas_shifted_fine.type())

        #print(alphas_shifted.shape)
        weights_fine = alphas_fine * torch.cumprod(alphas_shifted_fine, -1)[..., :-1] # (N_rays, N_samples_)
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        #print("weights_fine:",weights_fine.type())

        alphas_final_fine= torch.sum(weights_fine, -1)
        #print(weights_fine.shape, z_samp.shape)
        depth_final_fine = torch.sum(weights_fine * z_samp, -1) #/(alphas_final_fine+1e-8)  #/alphas_final_fine  # (B)
        depth_final_fine = torch.max(torch.min(depth_final_fine, z_far*torch.ones_like(depth_final_fine)), torch.zeros_like(depth_final_fine))

        #depth_final_fine  =  depth_final_fine.float()
        #depth_final_fine = torch.sum(weights_fine * z_samp, -1)  # (B)
        #-------------------------------------------------------------------------------------
        #print(weights_fine.shape, sample_all_fine.shape)

        feature_final_fine = torch.sum(weights_fine*sample_all_fine[:, :(Cube_channel-1), :, :, :], -1) # (N_rays, 3)
        del z_samp,deltas_z,deltas_z_fine, weights,alphas_shifted,sample_all_fine

        #print("feature_final_fine:",feature_final_fine.type())
        #print("alphas_final_fine:",alphas_final_fine.type())
        #feature_final_fine = torch.cat([depth_final_fine,feature_final_fine],dim=1)

        feature_final_fine = torch.cat([feature_final_fine,alphas_final_fine],dim=1)
        #del alphas
        depth_final_fine = self.upsampling(depth_final_fine)
        #m1 = nn.UpsamplingBilinear2d(scale_factor = 2)
        #
        #print("feature_final_fine:",feature_final_fine.type())
        conv_rgb_fine = self.upsampling(self.conv_up(feature_final_fine))
        #conv_rgb_fine = self.conv_up(feature_final_fine)


        feature_final_fine = self.upsampling(feature_final_fine)
        ray_rgb_fine = feature_final_fine[:,:3,:,:] #torch.Size([n_views, 3, 128, 128])
        alphas_final_fine = feature_final_fine[:,Cube_channel-1:,:,:]
        #print(feature_final_fine.dtype)
        feature_final_fine = feature_final_fine[:,:Cube_channel-1,:,:]
        feature_final_fine = feature_final_fine.permute(0, 2, 3, 1)

        #print(feature_final_coarse.dtype)
        #print("feature_final_fine:",feature_final_fine.type())

        #feature_final =  torch.cat([feature_final_fine,feature_final_coarse],dim=-1)
        #print("feature_final:",feature_final.type())

        #print(feature_final.dtype)
        #-------------------------------------------------------------------------------------
        #PK_tar = PK_tar.repeat(n_views,1,1,1)
        PK_tar = PK_tar.reshape(-1,PK_tar.shape[-1])
        PK_tar = self.Embedding_pk(PK_tar)
        #feature_final = feature_final.float()
        feature_final_fine = feature_final_fine.reshape(-1,feature_final_fine.shape[-1])
        feature_final_coarse = feature_final_coarse.reshape(-1,feature_final_coarse.shape[-1])
        #feature_final_coarse = feature_final_coarse.reshape(-1,feature_final_coarse.shape[-1]) 
        #feature_final_coarse = feature_final_coarse.repeat(1,2)
        #hidden = hidden.unsqueeze(1)
        #hidden = hidden.unsqueeze(1)
        #hidden = hidden.repeat(1,H,W,1)
        #hidden = hidden.reshape(-1,hidden.shape[-1])

        #print(feature_final_coarse.shape, feature_final.shape)
        #point_rgb_coarse = self.Mlp(Em_PK,feature_final_coarse,hidden)
        #feature_final = feature_final.astype(np.float32)
        #print(feature_final.shape)


        #print("feature_final:",PK_tar.shape, feature_final_coarse.shape, feature_final_fine.shape)

        #print("feature_final_fine:",PK_tar.shape, feature_final_coarse.shape)

        point_rgb_coarse = self.Mlp(PK_tar,feature_final_fine,feature_final_coarse)
        point_rgb_fine = point_rgb_coarse #self.Mlp(PK_tar,feature_final_fine)

        del camtrans_src,camdir_src,PK_tar,feature_final_fine,feature_final_coarse

        
        point_rgb_coarse = point_rgb_coarse.reshape(-1,H,W,3).permute(0, 3, 1, 2) #+ ray_rgb

        point_rgb_coarse = self.sigmoid(point_rgb_coarse + conv_rgb_coarse)

        point_rgb_fine = point_rgb_fine.reshape(-1,H,W,3).permute(0, 3, 1, 2) #+ ray_rgb

        #print("feature_final:",point_rgb_fine.shape, conv_rgb_fine.shape)

        point_rgb_fine = self.sigmoid(point_rgb_fine + conv_rgb_fine)
        #point_rgb = point_rgb + one_rayalpha - alphas_final
        #return point_rgb,ray_rgb,alphas_final
        return point_rgb_fine,point_rgb_coarse,ray_rgb_fine,ray_rgb_coarse,alphas_final_fine,alphas_final,depth_final_fine,depth_final_coarse

