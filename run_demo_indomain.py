import torch
import torch.nn as nn
from networks.generator import Generator_trans_w
import argparse
import numpy as np
import torchvision
from PIL import Image
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

        model_path = './checkpoint/in-domain440000.pt'
        
        print('==> loading model')
        self.gen =  Generator_trans_w(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        self.driving_path = args.driving_path

    def run(self, args):
        img_source = img_preprocessing(args.source_path, args.size).cuda()
        f_id = 0 # reference frame id
        
        vid_target, fps = vid_preprocessing(args.driving_path)
        vid_target = vid_target.cuda()
        print('==> running')
        with torch.no_grad():
            vid_target_recon = []

            for i in tqdm(range(vid_target.size(1))):
                vid_target_frame = self.gen.forward_relative(img_source, vid_target[:,i,:,:,:], vid_target[:,f_id,:,:,:])
                vid_target_recon.append(vid_target_frame.unsqueeze(2))

        vid_target_recon = torch.cat(vid_target_recon, dim=2)
        save_video(vid_target_recon, args.save_path, fps)
        print(f'Finish! Result video saving in {args.save_path}.')


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='source.jpg')
    parser.add_argument("--driving_path", type=str, default='input.mp4')
    parser.add_argument("--save_path", type=str, default='./output.mp4')
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    demo.run(args)
