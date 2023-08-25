# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Train an autoencoder."""
import argparse
import gc
import importlib
import importlib.util
import torch.nn as nn

import os, random, sys, time, imageio
#sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.utils.data
from data import get_split_dataset,SRNDataset,util
import torchvision.datasets as dsets
import models.neurvol_small_enr_dtu_ori_finedepth_depth_2mlp as aemodel
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary
import itertools
import torch.optim
import torchvision
from torchvision.utils import save_image
#torch.backends.cudnn.benchmark = True # gotta go fast!
torch.cuda.empty_cache()

class Logger(object):
    """Duplicates all stdout to a file."""
    def __init__(self, path, resume):
        #if not resume and os.path.exists(path):
        #    print(path + " exists")
        #    sys.exit(0)

        iternum = 0
        if resume:
            with open(path, "r") as f:
                for line in f.readlines():
                    match = re.search("Iteration (\d+).* ", line)
                    if match is not None:
                        it = int(match.group(1))
                        if it > iternum:
                            iternum = it
        self.iternum = iternum

        self.log = open(path, "a") if resume else open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
        

def get_loss_weights():
    #return {"irgbmse": 1.0, "kldiv": 0.001, "alphapr": 0.01, "tvl1": 0.01}
    return {"irgbmse": 1.0}

class AutomaticWeightedLoss(nn.Module):

 
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def gradient_x(img):
        gx = img[:,:,:,:-1] - img[:,:,:,1:]
        paddings = torch.zeros_like(gx[:,:,:,1])
        gx = torch.cat([gx,paddings.unsqueeze(-1)],-1)
        return gx

def gradient_y(img):
        gy = img[:,:,:-1,:] - img[:,:,1:,:]
        paddings = torch.zeros_like(gy[:,:,1,:])
        gy = torch.cat([gy,paddings.unsqueeze(-2)],2)
        return gy

def get_disparity_smoothness(disp, input_img):
        #disp_gradients_x = gradient_x(disp) #for d in disp]
        #disp_gradients_y = gradient_y(disp) #for d in disp]

        image_gradients_x = gradient_x(input_img) #for img in input_img]
        image_gradients_y = gradient_y(input_img) #for img in input_img]

        #weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        #weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]
        del input_img
        image_gradients_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        image_gradients_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        #smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        #smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        #smoothness_x = torch.sum(torch.abs(disp_gradients_x).mul(weights_x))/input_img.shape[0]
        #smoothness_y = torch.sum(torch.abs(disp_gradients_y).mul(weights_y))/input_img.shape[0]

        #smoothness =torch.mean(disp_gradients_x.mul(weights_x))
        #print(smoothness_x, smoothness_y)
        #del disp_gradients_x,disp_gradients_y,image_gradients_x,image_gradients_y,weights_x,weights_y

        #smoothness_x = [torch.nn.functional.pad(k,(0,1,0,0,0,0,0,0),mode='constant') for k in smoothness_x]
        #smoothness_y = [torch.nn.functional.pad(k,(0,0,0,1,0,0,0,0),mode='constant') for k in smoothness_y]

        return (torch.sum(torch.abs( gradient_x(disp)).mul(image_gradients_x))/disp.shape[0] + torch.sum(torch.abs(gradient_y(disp)).mul(image_gradients_y))/disp.shape[0])/119301

def train_criterion1(final_rgb_fine,final_rgb_coarse,ray_rgb_fine,ray_rgb_coarse,image_target,depth_final_fine, depth_final_coarse,n_views):
    criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    #criterion = torch.nn.L1Loss(reduce=True, size_average=True)

     #point_rgb_fine, point_rgb_fine,ray_rgb_fine,ray_rgb_coarse,alphas_final_fine,alphas_final

    #criterion = torch.nn.SmoothL1Loss(reduce=True, size_average=True)
    #loss1 = criterion(final_rgb,image_target)
    #image_target3 = image_target.repeat(n_views,1,1,1)
    #loss2 = criterion(middle_rgb,image_target3)
    #print("loss1:", float(loss1),"loss2:",float(loss2),"loss sum:",float(loss1+loss2))
    #awl = AutomaticWeightedLoss(2)
    #loss_sum = awl(loss1, loss2)

    #m = nn.LeakyReLU(-1)

    """
    alphas_final_fine = alphas_final_fine.reshape(alphas_final_fine.shape[0],-1)
    alphas_final_coarse = alphas_final_coarse.reshape(alphas_final_coarse.shape[0],-1)

    criterion1 = torch.log(0.1+ alphas_final_fine)+torch.log(1.1-alphas_final_fine)+ 2.20727 +torch.log(0.1+alphas_final_coarse)+torch.log(1.1-alphas_final_coarse)+ 2.20727

    #print(criterion1.grad_fn)
    
    #print(criterion1)
    criterion1 = torch.mean(torch.mean(criterion1, dim=-1))
    
    print(float(criterion1))
    
    depth_loss = criterionl1(depth_final_fine[:,:,:-1,:],depth_final_fine[:,:,1:,:])
    + criterionl1(depth_final_fine[:,:,:,:-1],depth_final_fine[:,:,:,1:])
    + criterionl1(depth_final_coarse[:,:,:-1,:],depth_final_coarse[:,:,1:,:])
    + criterionl1(depth_final_coarse[:,:,:,:-1],depth_final_coarse[:,:,:,1:])"""


    depth_loss = get_disparity_smoothness(depth_final_fine,image_target)+get_disparity_smoothness(depth_final_coarse,image_target)
    print(float(depth_loss)) 
    loss_sum = criterion(final_rgb,image_target)  + criterion(ray_rgb_coarse, image_target)/2+ criterion(ray_rgb_fine, image_target)/2 + depth_loss #+ 0.1*criterion1
    
    del depth_loss,final_rgb_fine,final_rgb_coarse,ray_rgb_fine,ray_rgb_coarse,image_target,depth_final_fine, depth_final_coarse
    


    #loss_sum = criterion(final_rgb_fine,image_target)  + criterion(final_rgb_coarse,image_target)  + criterion(ray_rgb_coarse, image_target)/2+ criterion(ray_rgb_fine, image_target)/2 + 0.1*criterion1
    #del image_target3
    #loss_sum = 0.5*criterion(middle_rgb,image_target) + 0.5*criterion(final_rgb,image_target) 
    #loss = criterion(final_rgb,image_target) + 0.1*criterion(final_sigma, mask) #+0.1(criterion(middle_rgb,image_target) + 0.1*criterion(middle_sigma, mask))
    return loss_sum


def train_criterion(final_rgb,middle_rgb,image_target,n_views):
    criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    #criterion = torch.nn.SmoothL1Loss(reduce=True, size_average=True)
    #loss1 = criterion(final_rgb,image_target)
    #image_target3 = image_target.repeat(n_views,1,1,1)
    #loss2 = criterion(middle_rgb,image_target3)

    #print("loss1:", float(loss1),"loss2:",float(loss2),"loss sum:",float(loss1+loss2))
    #awl = AutomaticWeightedLoss(2)
    #loss_sum = awl(loss1, loss2)
    loss_sum = criterion(final_rgb,image_target) + criterion(middle_rgb,image_target)
    #del image_target3
    #loss_sum = 0.5*criterion(middle_rgb,image_target) + 0.5*criterion(final_rgb,image_target) 
    #loss = criterion(final_rgb,image_target) + 0.1*criterion(final_sigma, mask) #+0.1(criterion(middle_rgb,image_target) + 0.1*criterion(middle_sigma, mask))
    return loss_sum


def get_rayalpha_loss(rayalpha):

    
    alphaprior = torch.mean(
                    torch.log(0.1 + rayalpha.view(rayalpha.size(0), -1)) +
                    torch.log(0.1 + 1. - rayalpha.view(rayalpha.size(0), -1)) - -2.20727, dim=-1)
    #alphaprior = torch.mean(torch.sin(rayalpha.view(rayalpha.size(0), -1)/0.314),dim=-1)
    #alphaprior = torch.mean(
    #                torch.log(0.1 + rayalpha.view(rayalpha.size(0), -1)) +
    #                torch.log(0.1 + 1. - rayalpha.view(rayalpha.size(0), -1)) , dim=-1)

    #print( alphaprior.shape)
    #print( alphaprior.shape)
    alphaprior = torch.mean(alphaprior)
    #print(alphaprior)
    #return {"irgbmse": 1.0, "kldiv": 0.001, "alphapr": 0.01, "tvl1": 0.01}
    return alphaprior

if __name__ == "__main__":
    # parse arguments
    """parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    #parser.add_argument('--profile', type=str, default="Train", help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.dirname(args.experconfig)
    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print(" ".join(sys.argv))
    print("Output path:", outpath)"""

    lr = 0.0001
    #dataset_format = "srn"
    dataset_format = "dvr_dtu" #dtu,dvr,srn
    su_format = "dtu"
    batch_size = 1
    split_batch_size = 4
    #N_views = 3
    #print("chair",os.path.basename(datadir))
    #is_chair = "chair" in dataset_name = os.path.basename(datadir)
    #print("chair",is_chair)
    is_igrida = True

    if is_igrida:

        datadir = "/gpfsscratch/rech/amg/uci85xy/Qian/Data/dtu_dataset/rs_dtu_4"
        outpath = "/gpfsscratch/rech/amg/uci85xy/Qian/Out"
        device_ids=[0,1,2,3]
        
        #datadir = "/srv/tempdd/qiali/data/chairs/black_chairs/chairs"
        #datadir = "/srv/tempdd/qiali/data/oneobj250/cars/yellow_cars/cars"
        #outpath = "/srv/tempdd/boukhaym/qiali"
        #datadir = "/srv/tempdd/boukhaym/qiali/data/dtu_dataset/rs_dtu_4"
        #datadir = "/gpfsscratch/rech/amg/uci85xy/Qian/Data/srn_chairs/chairs"
        #datadir = "/srv/tempdd/qiali/data/dtu_dataset/rs_dtu_4"
        #outpath = "/srv/tempdd/qiali"
        #datadir = "/srv/tempdd/qiali/data/oneobj250/cars/yellow_cars/cars"
        #outpath = "/srv/tempdd/qiali"
        #datadir ="/gpfsscratch/rech/amg/uci85xy/Qian/Data/oneobj250/cars/yellow_cars/cars"
        #datadir ="/gpfsscratch/rech/amg/uci85xy/Qian/Data/srn_cars/cars"
        #outpath ="/gpfsscratch/rech/amg/uci85xy/Qian/Out"
    else: 
       #datadir = "/home/qian/new/Sortdata/oneobj/chairs/black_chairs/chairs"
        #datadir = "/home/qian/new/Sortdata/srn_data/cars/cars"
        datadir = "/home/qian/new/Data/dtu_dataset/rs_dtu_4"
        #datadir = "/home/qian/new/Data/oneobj250/cars/yellow_cars/cars"
        outpath = "/home/qian/new/out"
        device_ids = [0,1]

    basename = os.path.basename(datadir)
    if dataset_format == "dvr":
        if su_format == "dtu":
            z_near=0.1
            z_far=5.0
        else:
            z_near=1.2
            z_far=4.0
    else:
        if os.path.basename(datadir) == "chairs":
            z_near = 1.25
            z_far = 2.75
        else:
            z_near = 0.8
            z_far = 1.8

    z_near = 0.1
    z_far= 5
    print(z_near,z_far)
    
    title = basename + "_enr_global_ori_finedepth_2mlp_64_128_32_depthloss_l2"
    outputimage_path =  outpath +"/out/"+ title
    outcheckpoint_path = outpath +"/checkpoints/" + title 
    os.makedirs(outputimage_path, exist_ok=True)
    os.makedirs(outcheckpoint_path,exist_ok=True)
    print("out image path:", outputimage_path)
    print(outcheckpoint_path)
    #os.makedirs(outcheckpoint_path, exist_ok=True)
    #print("out checkpoint path:", outcheckpoint_path)
     

    log = Logger("{}/checkpoints/log.txt".format(outpath), False)
    resume = False
    title1= basename + "_enr_global_rviews_newndc_newrendering"
    path_checkpoint =  outpath +"/checkpoints/" + title +"/"+title+"_30.pt"
    #datadir = "/home/qian/new/dataset/chairtwo/chairs"
    train_dataset, val_dataset, _ = get_split_dataset.get_split_dataset(dataset_format, datadir)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            #num_workers=8,
            pin_memory=False,
        )
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(batch_size, 1),
            shuffle=False,
            #num_workers=1,
            pin_memory=False,
        )

    print("loading data finished...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("device:",device)
    writer = SummaryWriter()
    ae = aemodel.Autoencoder()
    ae = torch.nn.DataParallel(ae, device_ids)
    ae = ae.to(device).train()

    start_epoch = 0
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr, betas=(0.9, 0.999))  
    load_pretrained = False
    pretrained_path = "/home/qian/new/checkpoints/checkpoints/cars/save/cars_randomviews_mulobj_enr_64rendering_64points_3.pt"

    #pretrained_path =  "/home/qian/new/checkpoints/checkpoints/chairs/chairs_randomviews_mulobj_enr_64rendering_64points_5.pt"
    #pretrained_path = outpath +"/checkpoints/"+ "cars_S1_32_128_21/cars_S1_32_128_21.pt"
    #pretrained_path = outpath +"/pretrained/cars/"+ "cars_S1_32_128_21.pt"
    #pretrained_path = outcheckpoint_path +"/"+title+"_1.pt"
    if load_pretrained:
        print("loading pretraind model: ", pretrained_path)
        pretrained_dict = torch.load(pretrained_path)['model_state_dict']  
        #pretrained_dict = torch.nn.DataParallel(pretrained_dict, device_ids=[0,1])
        ae_dict = ae.module.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in ae_dict}
        ae_dict.update(pretrained_dict) 
        #model = nn.DataParallel(model)
        ae.module.load_state_dict(ae_dict)
        del pretrained_dict

    if resume:
        checkpoint = torch.load(path_checkpoint)  
        ae.module.load_state_dict(checkpoint['model_state_dict']) 
        optimizer.load_state_dict(checkpoint['optim_state_dict'])  
        #train_criterion.load_state_dict(checkpoint['criterion_state_dict'])
        start_epoch = checkpoint['epoch']  
        print("start_epoch:",start_epoch)
        print('-----------------------------')
    new_start = 0 if start_epoch==-1 else start_epoch
  
     
    total_loss = 0
    iter = start_epoch

    print("start training...")

    for epoch in range(10000):

        for step_train, train_data in enumerate(train_loader):  
            #print("step_train:",step_train)
            #if step_train != 0:
            #    continue
            ae.train()

            all_images = train_data["image"]
            all_masks = train_data["masks"]
            #all_camrots = train_data["camrot"]
            #all_camtrans = train_data["campos"]
            all_focals = train_data["focal"]
            all_intrinsics = train_data["intrinsics"]
            #print("all_images,all_intrinsics:", all_images.shape, all_focals.shape, all_intrinsics.shape)
            #all_intrinsics /= 4
            #all_intrinsics[:,2,2] = 1
            all_poses = train_data["poses"]
            #all_poses = train_data["poses"]
            #all_pixelcoords = train_data["pixelcoords"]
            all_c = train_data["c"]
            #print("all_poses:",all_poses.shape)
            #print("all_c,focal:",all_c.shape, all_focals.shape)
            #print("allimage, pixelcoord", all_images.shape, all_pixelcoords.shape)
            all_camrots = all_poses[:,:,:3,:3]
            all_camtrans = all_poses[:,:,:3,3]

            #print("all_intrinsics:",all_intrinsics.shape)

            SB, NV, _, H, W = all_images.shape


            #print(all_images.shape)
            for obj_idx in range(SB):
                
                images = all_images[obj_idx]  # (NV, 3, H, W)
                camrots = all_camrots[obj_idx]  
                camtrans = all_camtrans[obj_idx]
                focals = all_focals[obj_idx]
                masks = all_masks[obj_idx]

                #focals = focals/4
                intrinsics = all_intrinsics[obj_idx].unsqueeze(0)
                intrinsics = intrinsics.expand(images.shape[0],-1,-1)

                #print("intrinsics", intrinsics.shape)
                #pixelcoord = all_pixelcoords[obj_idx].to(device=device)
                poses = all_poses[obj_idx]
                #focal1 = focals[1,0]
                c = all_c[obj_idx]
                #print("focal:",focals.shape)
                #print("focal:",focals)
                #print("c:",c)
                #W = int(W)
                #H = int(H)
                #c =c/4
                #princpts1 = princpts[1,0]
                #print(focals.shape,princpts.shape )
                cam_dirs = util.gen_rays(
                    poses, W, H, focal = focals, z_near = z_near, z_far = z_far, c=c
                )   
                print("train f,c:",focals,c)
                #print(cam_rays.shape)
                #poses = all_poses[obj_idx]
                #print("pixelcoord", pixelcoord.shape)
                #print("images.shape",images.shape)
               
                #print(split_batch_size)
                #print("all shapes:",camrots.shape, camposs.shape, focals.shape, princpts.shape,pixelcoords.shape)
                #each_batch_size = min(50,NV)
                #------------------------------------------------------
                #print("image:",images.shape)
                #for target_image in images:
                #    print("image:",target_image.shape)


                split_image_target = torch.split(images, split_batch_size,dim=0)
                split_mask_target = torch.split(masks, split_batch_size,dim=0)
                split_camrot_target = torch.split(camrots, split_batch_size,dim=0)
                split_camtran_target = torch.split(camtrans, split_batch_size,dim=0)
                #split_focal_input = torch.split(focals, split_batch_size,dim=0)
                split_intrinsic_target = torch.split(intrinsics, split_batch_size,dim=0)
                #split_pixelcoord_input = torch.split(pixelcoords, split_batch_size,dim=0)
                #split_pose_input = torch.split(poses, split_batch_size,dim=0)
                split_camdir_target = torch.split(cam_dirs, split_batch_size,dim=0)
                
                #print("image_target shape:", images.shape,camrots.shape,camtrans.shape, cam_dirs.shape) #([50, 3, 128, 128])
               
                for image_target, camrot_target, camtran_target,camdir_target, intrinsic_target, mask_target in zip(
                    split_image_target, split_camrot_target, split_camtran_target,split_camdir_target,split_intrinsic_target, split_mask_target):
                #for image_target, camrot_target, camtran_target,camdir_target, intrinsic_target in zip(
                #    images, camrots, camtrans,intrinsics,cam_dirs):
                    #intrinsic = intrinsic.unsqueeze(0).repeat(image_target.)
                    #N_views = random.randint(1,3)
                    N_views = 3
                    all_src_images = []
                    all_src_camtrans = []
                    all_src_camrots = []
                    all_src_camdirs = []

                    #new_all_src_images = []

                    #all_src_intrinsics = []
                    #print(":", all_camtrans.shape, all_camrots.shape)
                    for i in range(image_target.shape[0]):
                        src_idx = random.sample(range(0, NV-1), N_views)   
                        #print("src_idxs:", src_idxs)
                        #new_src_image = images[src_idxs]

                        #src_idx = random.randint(0,NV-1)
                        src_image = images[src_idx]
                        src_camtran = camtrans[src_idx]
                        src_camrot = camrots[src_idx]
                        src_camdir = cam_dirs[src_idx]
                        #src_intrinsic = intrinsics

                        #new_all_src_images.append(new_src_image)
                        all_src_images.append(src_image)
                        all_src_camtrans.append(src_camtran)
                        all_src_camrots.append(src_camrot)
                        all_src_camdirs.append(src_camdir)
                        #all_src_intrinsics.append(src_intrinsic)
                        #all_src_poses.append(src_pose)
                    del src_image
                    del src_camtran
                    del src_camrot
                    del src_camdir
                    #del src_intrinsic

                    #new_all_src_images = torch.stack(new_all_src_images)

                    all_src_images = torch.stack(all_src_images)
                    all_src_camtrans = torch.stack(all_src_camtrans)
                    all_src_camrots = torch.stack(all_src_camrots)
                    all_src_camdirs = torch.stack(all_src_camdirs)
                    all_src_images = (all_src_images*0.5+0.5) #.to(device=device)

                    image_target = (image_target*0.5+0.5) #.to(device=device)
                    
                    #print(":",camrot_target.shape, camtran_target.shape, camdir_target.shape, all_src_camrots.shape, all_src_camtrans.shape, all_src_camdirs.shape )
                    final_rgb,final_rgb_coarse,ray_rgb_fine,ray_rgb_coarse,alphas_final_fine, alphas_final,depth_final_fine,depth_final_coarse = ae(n_views = N_views, camrot_tar = camrot_target, camtrans_tar = camtran_target, 
                        camdir_tar = camdir_target, intrinsic_tar = intrinsic_target, image_src = all_src_images, camrot_src = all_src_camrots, 
                        camtrans_src=all_src_camtrans,camdir_src = all_src_camdirs, z_near = z_near, z_far = z_far)
                    #def forward(self, losslist, image_tar=None,camrot_tar, camtrans_tar, camray_tar = None, focal, princpt,image_src = None, camrot_src = None, camtrans_src=None,camray_src = None, viewtemplate=False,outputlist=[]):
                    #point_rgb_fine, point_rgb_fine,ray_rgb_fine,ray_rgb_coarse,alphas_final_fine,alphas_final
                    #output = ae(iternum=iternum,losslist=lossweights.keys(),camrot = camrots,campos = camposs,
                    #   focal = focals,princpt = princpts,pixelcoord = pixelcoord,
                    #   validinput = None,image_tar=images, image_src = all_src_images, imagevalid=None, viewtemplate=False)

                    #print("data.shape, output.shape:",depth_final_fine.shape, depth_final_coarse.shape)
                    # compute final loss
                    #print("losses.items():", output["losses"].items())
                    optimizer.zero_grad()
                    mask_target = mask_target.to(final_rgb.device)
                    image_target = image_target.to(final_rgb.device)
                    #print("images shape:",final_rgb.shape,feature.shape, image_target.shape )
                    #loss_fn = torch.nn.MSELoss(reduce= True, size_average = False)
                    #
                    loss = train_criterion1(final_rgb,final_rgb_coarse,ray_rgb_fine,ray_rgb_coarse,image_target,
                        depth_final_fine,depth_final_coarse,N_views)
                    #alpha_loss = get_rayalpha_loss(alpha_1)
                    #loss = image_loss #+ 0.1*alpha_loss
                    #target = torch.ones_like(output_alphas)
                    #loss = criterion(output,image_target) #+ 0.01*get_rayalpha_loss(output_alphas)
                    #print(loss)
                    #if epoch !=0 and epoch % 10 == 0:
                    #if (step_train %20 == 0):
                    
                    if ( step_train %50 == 0):
                        all_src_images = all_src_images.to(final_rgb.device)
                        #final_sigma = final_sigma.reshape(-1,3,H,W)
                        """if N_views == 1:
                            vis = torch.stack((all_src_images[:,0,:,:,:], final_sigma,final_rgb,image_target),dim=1)
                            nrows = 4
                            for i in range(vis.shape[0]):
                                image_batch = torchvision.utils.make_grid(vis[i,:,:,:,:],nrows)
                                save_image(image_batch, os.path.join(outputimage_path, 'train_img_%d_%d_%.4f.png'%(epoch+start_epoch,step_train,float(loss))))
                        if N_views == 2:
                            vis = torch.stack((all_src_images[:,0,:,:,:],all_src_images[:,1,:,:,:],final_sigma,final_rgb,image_target),dim=1)
                            nrows = 5
                            for i in range(vis.shape[0]):
                                image_batch = torchvision.utils.make_grid(vis[i,:,:,:,:],nrows)
                                save_image(image_batch, os.path.join(outputimage_path, 'train_img_%d_%d_%.4f.png'%(epoch+start_epoch,step_train,float(loss))))
                        if N_views == 3:"""
                        depth_final_coarse = (depth_final_coarse-torch.min(depth_final_coarse))/(torch.max(depth_final_coarse)-torch.min(depth_final_coarse)+1e-8)
                        depth_final_fine = (depth_final_fine-torch.min(depth_final_fine))/(torch.max(depth_final_fine)-torch.min(depth_final_fine)+1e-8)

                        nrows = 3
                        vis = torch.stack((all_src_images[:,0,:,:,:],all_src_images[:,1,:,:,:],all_src_images[:,2,:,:,:],
                            ray_rgb_coarse,ray_rgb_fine,image_target,
                            final_rgb_coarse,final_rgb,
                            image_target,depth_final_fine.repeat(1,3,1,1),depth_final_coarse.repeat(1,3,1,1),
                             alphas_final.repeat(1,3,1,1),alphas_final_fine.repeat(1,3,1,1),mask_target.repeat(1,3,1,1)),dim=1)
                        for i in range(vis.shape[0]):
                            image_batch = torchvision.utils.make_grid(vis[i,:,:,:,:],nrows)
                            save_image(image_batch, os.path.join(outputimage_path, 'train_img_%d_%d_%.4f.png'%(epoch+start_epoch,step_train,float(loss))))
                        del vis,depth_final_coarse,depth_final_fine
                            
                    #print(new_loss)
                    #loss = criterion(output,image_target)  + 0.01*get_rayalpha_loss(output_alphas)
                    #loss = criterion(output,image_target)  + 0.1*F.binary_cross_entropy_with_logits(output_alphas,target,reduction='mean')
                    #loss = (image_tar - output) ** 2
                    #loss = sum(torch.sum(k)for k,v in output["losses"]["irgbmse"])/images.shape[0]
                    loss.backward()
                    optimizer.step()
                    writer.add_scalar(title +"train_loss", loss, iter)
                    #total_loss += float(loss)
                    #step_train += 1
                    #if epoch != 0 and epoch % 200 == 0:
                        #img05 = 510*(output[0] -0.5)
                    #    save_image(output[0], os.path.join(outputimage_dir, 'train_img%d_%.4f_0.png'%(epoch,float(loss))))
                        #save_image(img1, os.path.join(outputimage_dir, 'img%d_%d_%.4f_imgout.png'%(epoch,step_val,float(val_loss))))
                        #save_image(img05, os.path.join(outputimage_dir, 'train_img%d_%.4f_05.png'%(epoch,float(loss))))
                    iter = iter + 1


                    print("train_epoch:", epoch+start_epoch, "step_train:", step_train, N_views, title, "loss:",float(loss))
                    del loss


                #b = train_data["campos"].size(0)
                #writer.batch(iternum, iternum * Train.batchsize + torch.arange(b), **testbatch, **testoutput)
            #if loss.item() > 20 * prevloss or not np.isfinite(loss.item()):
            #    print("Unstable loss function; resetting")

                #ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)

                #prevloss = loss.item()

                # compute evaluation output
            #if iternum% 1000 == 0:
            #if epoch != 0 and epoch%10 ==0:

            if ( step_train %50 == 0):
            #if (step_train%50 == 0):
                ae.eval()
                with torch.no_grad():
                    for step_val, val_data in enumerate(val_loader):  
                        #N_views = random.randint(1,3)
                        N_views = 3
                        #N_views = 64
                        #if step_val > 50: 
                        #    continue
                        val_images = val_data["image"] #.to(device=device)
                        val_masks = val_data["masks"] #.to(device=device)
                        #val_camrots = val_data["camrot"].to(device=device)
                        #val_camtrans = val_data["campos"].to(device=device)
                        val_focals = val_data["focal"] #.to(device=device)
                        all_c = val_data["c"] #.to(device=device)
                        val_intrinsics = val_data["intrinsics"]  #.to(device=device)
                        #val_princpts = val_data["princpt"].to(device=device)
                        #val_pixelcoords = val_data["pixelcoords"].to(device=device)
                        val_poses = val_data["poses"] #.to(device=device)
                        val_camrots = val_poses[:,:,:3,:3]
                        val_camtrans = val_poses[:,:,:3,3]
                        Num_obj_val, Num_view_val, _, H, W = val_images.shape
               
                        if Num_obj_val == 1:
                            obj_idx_val = 0
                        else:
                            obj_idx_val = random.randint(0,Num_obj_val-1)

                        #view_idx_src_val = random.sample(range(0, Num_view_val-1), N_views)   
                   
                        view_idx_src_val = [22,25,28]
                        #view_idx_src_val = random.sample(range(0, Num_view_val-1), N_views)   
                        test_list = [1,2,8,9,10,11,12,13,14,15,23,24,26,27,29,30,31,32,33,34,35,36,40,41,42,43,44,45,46,47,48,49]
                        view_idx_index = random.randint(0,30)
                        view_idx_tar_val =  test_list[view_idx_index]

                        image_src = val_images[obj_idx_val][view_idx_src_val].unsqueeze(0)  # (NV, 3, H, W)
                        image_src = (image_src*0.5+0.5) #.to(device=device)

                        pose_src = val_poses[obj_idx_val]#[view_idx_src_val].unsqueeze(0)
                        camrot_src = val_camrots[obj_idx_val][view_idx_src_val].unsqueeze(0)  # (NV, 3, H, W)
                        camtran_src = val_camtrans[obj_idx_val][view_idx_src_val].unsqueeze(0)  # (NV, 3, H, W)
                        focal = val_focals[obj_idx_val].unsqueeze(0)
                        #W = int(W/4)
                        #H = int(H/4)
                        c = all_c[obj_idx_val]
                        camdirs_src = util.gen_rays(
                            pose_src, W, H, focal = focal, z_near = z_near, z_far = z_far, c=c
                        )
                        #print("camdirs_src:", pose_src.shape,camdirs_src.shape )
                        pose_src = pose_src[view_idx_src_val].unsqueeze(0)
                       
                        camdirs_src = camdirs_src[view_idx_src_val].unsqueeze(0)
                        #print("image_src:", image_src.shape, pose_src.shape, camrot_src.shape, camtran_src.shape)
                        image_target = (val_images[obj_idx_val][view_idx_tar_val].unsqueeze(0)*0.5+0.5) #.to(device=device)  # (NV, 3, H, W)
                        mask_target = val_masks[obj_idx_val][view_idx_tar_val].unsqueeze(0) #.to(device=device)  # (NV, 3, H, W)
                        camrot_target = val_camrots[obj_idx_val][view_idx_tar_val].unsqueeze(0)
                        camtran_target = val_camtrans[obj_idx_val][view_idx_tar_val].unsqueeze(0)
                        intrinsic_target = val_intrinsics[obj_idx_val].unsqueeze(0)
                        pose_tar = val_poses[obj_idx_val][view_idx_tar_val].unsqueeze(0)
                        camdir_target = util.gen_rays(
                            pose_tar, W, H, focal = focal, z_near = z_near, z_far = z_far, c=c
                        )  

                        #intrinsics = intrinsics.expand(image_target.shape[0],-1,-1)
                        #print("intrinsic_target:",intrinsic_target.shape)
                        #focal = focal/4
                        #print("val_images.shape, val_focals.shape:", image_target.shape, camrot_target.shape,camtran_target.shape, focal.shape, intrinsic_target.shape)

                        print("val c,focal,",c,focal)
                        #princpt = val_princpts[obj_idx_val][view_idx_tar_val].unsqueeze(0)
                        #pixelcoord = val_pixelcoords[obj_idx_val]
                        #print("val_pixelcoord:", pixelcoord.shape)
                        
                        
                        final_rgb,final_rgb_coarse,ray_rgb_fine,ray_rgb_coarse,alphas_final_fine, alphas_final,depth_final_fine,depth_final_coarse  = ae(n_views = N_views, camrot_tar = camrot_target, camtrans_tar = camtran_target, 
                            camdir_tar = camdir_target, intrinsic_tar = intrinsic_target, image_src = image_src, camrot_src = camrot_src, 
                            camtrans_src = camtran_src,camdir_src = camdirs_src, z_near = z_near, z_far = z_far,)
                        #lossitem = output["losses"] 
                        #print("lossitem", lossitem ["irgbmse"][0],lossitem ["irgbmse"][1])
                        #print(lossitem ["irgbmse"][0])
                        #test_loss = sum(torch.sum(k)for k in testoutput["losses"]["irgbmse"])
                        #val_loss = criterion(testoutput,image_target)
                        #target = torch.ones_like(test_alapha)
                        #val_loss = criterion(testoutput,image_target) # + 0.1*F.binary_cross_entropy_with_logits(test_alapha,target,reduction='mean')
                        #val_loss = 0.8*criterion(final_rgb,image_target) + 0.2*criterion(rendering_rgb,image_target)
                        mask_target = mask_target.to(final_rgb.device)
                        image_target = image_target.to(final_rgb.device)
                        #print(final_sigma.shape, final_rgb.shape,image_target.shape)
                        val_loss = train_criterion1(final_rgb,final_rgb_coarse,ray_rgb_fine,ray_rgb_coarse,image_target,depth_final_fine,depth_final_coarse,N_views)
                        image_src = image_src.to(final_rgb.device)
                   
                        #psnr0 = util.psnr(ray_rgb, image_target)
                        psnr = util.psnr(final_rgb, image_target)
                        #imgout = np.clip(np.clip(testoutput[0] / 255., 0., 255.) ** (1. / 1.8) * 255., 0., 255).astype(np.uint8)
                        #save_image(ray_rgb, os.path.join(outputimage_path, 'img_stage1_%d_%d_%.4f_2.png'%(epoch+start_epoch ,step_val,float(val_loss))))
                        #save_image(final_rgb, os.path.join(outputimage_path, 'img_stage2_%d_%d_%.4f_2.png'%(epoch+start_epoch ,step_val,float(val_loss))))
                        #save_image(alpha, os.path.join(outputimage_path, 'img_alpha_%d_%d_%.4f_2.png'%(epoch+start_epoch,step_train,float(loss))))

                        #print(final_sigma.shape, final_rgb.shape)
                        #nrows = 1
                        image_src = image_src.to(final_rgb.device)
                        depth_final_coarse = (depth_final_coarse-torch.min(depth_final_coarse))/(torch.max(depth_final_coarse)-torch.min(depth_final_coarse)+1e-8)
                        depth_final_fine = (depth_final_fine-torch.min(depth_final_fine))/(torch.max(depth_final_fine)-torch.min(depth_final_fine)+1e-8)

                        #print(final_sigma.shape, final_rgb.shape)
                        #if N_views == 1:
                        vis = torch.stack((image_src[:,0,:,:,:], image_src[:,1,:,:,:],image_src[:,2,:,:,:],
                            ray_rgb_coarse,ray_rgb_fine,image_target,final_rgb_coarse,final_rgb,
                            image_target,depth_final_fine.repeat(1,3,1,1),depth_final_coarse.repeat(1,3,1,1),
                            alphas_final.repeat(1,3,1,1),alphas_final_fine.repeat(1,3,1,1),mask_target.repeat(1,3,1,1)),dim=1)
                        for i in range(vis.shape[0]):
                            image_batch = torchvision.utils.make_grid(vis[i,:,:,:,:],nrows)
                            save_image(image_batch, os.path.join(outputimage_path, 'val_img_%d_%d_%.4f_max.png'%(epoch+start_epoch,step_train,float(val_loss))))
                        del vis
                        print("---val_epoch---:", epoch+start_epoch, "step:", step_val, title, "N_views:", N_views, "loss:",float(val_loss),"psnr:",psnr)
                        step_val += 1
                        writer.add_scalar(title + "val" , val_loss, iter)
                        writer.add_scalar(title + "psnr", psnr, iter)
                        del val_loss,depth_final_coarse,depth_final_fine
                        #writer.add_scalar(title + "psnr", psnr0, iter)

                        #if(step_val>0):
                        #    break

                        #testoutput = ae(iternum, [], **{k: x.to("cuda") for k, x in testbatch.items()}, **Render.get_ae_args())
            
            # save intermediate results
            #if epoch != 0  and epoch % 50 == 0:
        if epoch+start_epoch >10 and epoch % 2 == 0:

            checkpoint_dict = {'epoch': epoch+start_epoch, 
                   'model_state_dict': ae.module.state_dict(), 
                   'optim_state_dict': optimizer.state_dict(), }
                   #'criterion_state_dict': train_criterion.state_dict()
            torch.save(checkpoint_dict, outcheckpoint_path+"/"+title+"_%d.pt"%(epoch+start_epoch))
        writer.flush()

        # cleanup
        #writer.finalize()
