import torch
import torch.nn as nn
from networks.generator import Generator_trans_w
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import Resize

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2)
    torch_resize = Resize([256,256])
    vid = torch_resize(vid).unsqueeze(0)

    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps

def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        model_path = './checkpoint/cross-domain.pt'

        print('==> loading model')
        self.gen =  Generator_trans_w(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()
        self.gen_real =  Generator_trans_w(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen_real']
        self.gen_real.load_state_dict(weight)
        self.gen_real.eval()

        print('==> loading data')
        self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        self.driving_path = args.driving_path

    def encode_motion(self, img, s=False):
        bs,_,_,_ = img.shape
        query_0 = self.gen.dec.direction.weight.unsqueeze(1).repeat(1, bs, 1)
        feat, _ = self.gen.enc2(img)
        if s:
            return self.gen_real.dec.transformerlayer_0(query_0, feat.unsqueeze(0), feat.unsqueeze(0))
        else:
            return self.gen_real.dec.transformerlayer_1(query_0, feat.unsqueeze(0), feat.unsqueeze(0)) 

    def run_r2c(self, args):
        
        img_source = img_preprocessing(args.source_path, args.size).cuda()
        f_id = 0
        vid_target, fps = vid_preprocessing(args.driving_path)
        vid_target = vid_target.cuda()
        print('==> running')
        with torch.no_grad():
            vid_target_recon = []
            real_motion_source0 = self.gen_real.encode_motion(vid_target[:,f_id,:,:,:])
            cartoon_s, cartoon_source_feat = self.gen.enc(img_source)
            cartoon_motion_source = self.encode_motion(img_source, True)
            cartoon_motion_source0 = self.encode_motion(img_source)
            for i in range(vid_target.size(1)):
                img_target = vid_target[:, i, :, :, :]    
                real_motion_drive = self.gen_real.encode_motion(img_target)
                vid_target_frame = self.gen.dec.forward_relative(cartoon_s, cartoon_motion_source,real_motion_drive,real_motion_source0,cartoon_motion_source0,cartoon_source_feat)
                vid_target_recon.append(vid_target_frame.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            save_video(vid_target_recon, args.save_path, fps)
            print(f'Finish! Result video saving in {args.save_path}.')

    def run_c2r(self, args):
        img_source = img_preprocessing(args.source_path, args.size).cuda()
        f_id = 0
        vid_target, fps = vid_preprocessing(args.driving_path)
        vid_target = vid_target.cuda()
        print('==> running')
        with torch.no_grad():
            vid_target_recon = []
            cartoon_motion_source0 = self.encode_motion(self.vid_target[:,f_id,:,:,:])
            real_s, real_source_feat = self.gen_real.enc(img_source)
            real_motion_source = self.gen_real.encode_motion(img_source, True)
            real_motion_source0 = self.gen_real.encode_motion(img_source)
            for i in range(vid_target.size(1)):
                img_target = vid_target[:, i, :, :, :]   
                cartoon_motion_drive = self.encode_motion(img_target)
                vid_target_frame = self.gen_real.dec.forward_relative(real_s, real_motion_source, cartoon_motion_drive, cartoon_motion_source0, real_motion_source0, real_source_feat)
                vid_target_recon.append(vid_target_frame.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            save_video(vid_target_recon, args.save_path, fps)
            print(f'Finish! Result video saving in {args.save_path}.')

if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--type", type=str, choices=['c2r', 'r2c'], default='c2r')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='source.jpg')
    parser.add_argument("--driving_path", type=str, default='input.mp4')
    parser.add_argument("--save_path", type=str, default='./output.mp4')
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    if args.type == 'c2r':
        demo.run_c2r(args)
    elif args.type == 'r2c':
        demo.run_r2c(args)
    else:
        raise NotImplementedError
