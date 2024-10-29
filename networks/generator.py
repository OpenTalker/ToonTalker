from torch import nn
from .encoder import EncoderApp3, EncoderApp
from .styledecoder import Synthesis_transformer_SFT
import torch

class Generator_trans_w(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_trans_w, self).__init__()

        self.enc = EncoderApp3(size, style_dim)
        self.enc2 = EncoderApp(size, style_dim)
        self.dec = Synthesis_transformer_SFT(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    def encode_motion(self, img, is_source = False):
        bs,_,_,_ = img.shape
        query_0 = self.dec.direction.weight.unsqueeze(1).repeat(1, bs, 1)
        feat, _ = self.enc2(img)
        if is_source:
            return self.dec.transformerlayer_0(query_0, feat.unsqueeze(0), feat.unsqueeze(0))
        else:
            return self.dec.transformerlayer_1(query_0, feat.unsqueeze(0), feat.unsqueeze(0))

    def forward(self, img_source, img_drive):
        res_s, feats = self.enc(img_source)
        source, _ = self.enc2(img_source)
        drive, _ = self.enc2(img_drive)
        img_recon = self.dec(res_s, source.unsqueeze(0), drive.unsqueeze(0), feats)
        return img_recon
    
    def forward_relative(self, img_source, img_drive, start):
        res_s, feats = self.enc(img_source)
        source, _ = self.enc2(img_source)
        drive, _ = self.enc2(img_drive)
        h_start, _ = self.enc2(start)
        img_recon = self.dec.forward_relative(res_s, source.unsqueeze(0), drive.unsqueeze(0), feats, h_start.unsqueeze(0))
        return img_recon