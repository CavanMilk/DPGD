import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import make_coord
from models.edsr import make_edsr_baseline


class Coupled_Layer(nn.Module):
    def __init__(self,
                 coupled_number=32,
                 n_feats=64,
                 kernel_size=3):
        super(Coupled_Layer, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size
        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_rgb_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_rgb_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

    def forward(self, feat_dlr, feat_rgb):
        shortCut = feat_dlr
        feat_dlr = F.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
                            padding=1)
        feat_dlr = F.relu(feat_dlr, inplace=True)
        feat_dlr = F.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
                            padding=1)
        feat_dlr = F.relu(feat_dlr + shortCut, inplace=True)
        shortCut = feat_rgb
        feat_rgb = F.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
                            padding=1)
        feat_rgb = F.relu(feat_rgb, inplace=True)
        feat_rgb = F.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
                            padding=1)
        feat_rgb = F.relu(feat_rgb + shortCut, inplace=True)
        return feat_dlr, feat_rgb


class Coupled_Encoder(nn.Module):
    def __init__(self,
                 n_feat=64,
                 n_layer=4):
        super(Coupled_Encoder, self).__init__()
        self.n_layer = n_layer
        self.init_deep = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
        )
        self.init_rgb = nn.Sequential(
            nn.Conv2d(3, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
        )
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer() for i in range(self.n_layer)])

    def forward(self, feat_dlr, feat_rgb):
        feat_dlr = self.init_deep(feat_dlr)
        feat_rgb = self.init_rgb(feat_rgb)
        for layer in self.coupled_feat_extractor:
            feat_dlr, feat_rgb = layer(feat_dlr, feat_rgb)
        return feat_dlr, feat_rgb


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class JIIF_Cou(nn.Module):

    def __init__(self, args, feat_dim=128, guide_dim=128, mlp_dim=[1024,512,256,128]):
        super().__init__()
        self.args = args
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim

        self.image_encoder = make_edsr_baseline(n_feats=self.guide_dim, n_colors=3)
        self.depth_encoder = make_edsr_baseline(n_feats=self.feat_dim, n_colors=1)
        self.Couple_Encode=Coupled_Encoder()

        imnet_in_dim = self.feat_dim + self.guide_dim * 2 + 2
        
        self.imnet = MLP(imnet_in_dim, out_dim=2, hidden_list=self.mlp_dim)
        
    def query(self, feat, coord, hr_guide, lr_guide, image):
        
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W
          
        b, c, h, w = feat.shape # lr
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []
        
        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx 
                coord_[:, :, 1] += (vy) * ry 
                k += 1

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                q_guide_lr = F.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [B, N, C]
                q_guide = torch.cat([q_guide_hr, q_guide_hr - q_guide_lr], dim=-1)

                inp = torch.cat([q_feat, q_guide, rel_coord], dim=-1)

                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1) # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1) # [B, N, 2, kk]
        weight = F.softmax(preds[:,:,1,:], dim=-1)

        ret = (preds[:,:,0,:] * weight).sum(-1, keepdim=True)

        return ret

    def forward(self, data):
        image, depth, coord, res, lr_image = data['image'], data['lr'], data['hr_coord'], data['lr_pixel'], data['lr_image']

        hr_guide,lr_guide=self.Couple_Encode.forward(lr_image,image)
        # hr_guide = self.image_encoder(image)
        # lr_guide = self.image_encoder(lr_image)

        feat = self.depth_encoder(depth)

        if self.training or not self.args.batched_eval:
            res = res + self.query(feat, coord, hr_guide, lr_guide, data['hr_depth'].repeat(1,3,1,1))

        # batched evaluation to avoid OOM
        else:
            N = coord.shape[1] # coord ~ [B, N, 2]
            n = 30720
            tmp = []
            for start in range(0, N, n):
                end = min(N, start + n)
                ans = self.query(feat, coord[:, start:end], hr_guide, lr_guide, data['hr_depth'].repeat(1,3,1,1)) # [B, N, 1]
                tmp.append(ans)
            res = res + torch.cat(tmp, dim=1)

        return res


