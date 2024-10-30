import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import make_coord
from models.edsr import make_edsr_baseline
idx = 0

class MDTA(nn.Module):
    def __init__(self, channel, num_head):
        super(MDTA, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(1, num_head, 1, 1))
        self.qkv = nn.Conv2d(channel, channel*3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channel*3, channel*3, kernel_size=3, padding=1, groups=channel*3, bias=False)
        self.project_out = nn.Conv2d(channel, channel, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_head, -1, h*w)
        k = k.reshape(b, self.num_head, -1, h*w)
        v = v.reshape(b, self.num_head, -1, h*w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous())*self.temperature, dim=1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN (nn.Module):
    def __init__(self, channel, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channel = int(channel*expansion_factor)
        self.project_in = nn.Conv2d(channel, hidden_channel*2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channel*2, hidden_channel*2, kernel_size=3, padding=1,
                              groups=hidden_channel*2, bias=False)
        self.project_out = nn.Conv2d(hidden_channel, channel, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1)*x2)
        return x


class TransFormerBlock(nn.Module):
    def __init__(self, channel, num_head, expansion_factor=2.66):
        super(TransFormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channel)
        self.attn = MDTA(channel, num_head)
        self.norm2 = nn.LayerNorm(channel)
        self.channel = channel
        self.ffn = FeedForward(channel, expansion_factor, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        #归一
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w))
        return x


class Coupled_Layer(nn.Module):
    def __init__(self, num_head,
                 coupled_number=32,
                 n_feats=128,
                 kernel_size=3,):
        super(Coupled_Layer, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size
        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[(self.n_feats - self.coupled_number), self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[(self.n_feats - self.coupled_number), self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[(self.n_feats - self.coupled_number), self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[(self.n_feats - self.coupled_number), self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_1 = nn.Parameter((torch.zeros(size=[(self.n_feats - self.coupled_number)])))
        self.bias_rgb_1 = nn.Parameter((torch.zeros(size=[(self.n_feats - self.coupled_number)])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_2 = nn.Parameter(( torch.zeros(size=[(self.n_feats - self.coupled_number)])))
        self.bias_rgb_2 = nn.Parameter((torch.zeros(size=[(self.n_feats - self.coupled_number)])))
        self.attention1 = TransFormerBlock(n_feats, num_head)
        self.attention2 = TransFormerBlock(n_feats, num_head)
        self.attention3 = TransFormerBlock(n_feats, num_head)
        self.attention4 = TransFormerBlock(n_feats, num_head)

    def forward(self, feat_dlr, feat_rgb):
        global idx
        shortCut = feat_dlr
        feat_dlr = F.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
                            padding=1)
        feat_dlr = self.attention1(feat_dlr)
        feat_dlr = F.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
                            padding=1)
        feat_dlr = self.attention2(feat_dlr)
        feat_dlr = F.relu(feat_dlr + shortCut, inplace=True)
        shortCut = feat_rgb
        feat_rgb = F.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
                            padding=1)
        feat_rgb = self.attention3(feat_rgb)
        feat_rgb = F.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
                            padding=1)
        feat_rgb = self.attention4(feat_rgb)
        feat_rgb = F.relu(feat_rgb + shortCut, inplace=True)
        return feat_dlr, feat_rgb


class Coupled_Encoder(nn.Module):
    def __init__(self, num_head,
                 n_feat=128,
                 n_layer=4):
        super(Coupled_Encoder, self).__init__()
        self.n_layer = n_layer
        self.init_deep = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            TransFormerBlock(n_feat, 4, expansion_factor=2.66)
        )
        self.init_rgb = nn.Sequential(
            nn.Conv2d(3, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            TransFormerBlock(n_feat, 4, expansion_factor=2.66)
        )
        self.init_edge = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            TransFormerBlock(n_feat, 4, expansion_factor=2.66)
        )
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer(num_head=num_head) for i in range(self.n_layer)])

    def forward(self, feat_dlr, feat_rgb, edge):
        feat_dlr = self.init_deep(feat_dlr)
        feat_rgb = self.init_rgb(feat_rgb)
        feat_edge = self.init_edge(edge)
        feat_dlr = feat_dlr + feat_edge
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


class GPGD(nn.Module):

    def __init__(self, args, feat_dim=128, guide_dim=128, mlp_dim=[1024,512,256,128]):
        super().__init__()
        self.args = args
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim

        self.image_encoder = make_edsr_baseline(n_feats=self.guide_dim, n_colors=3)
        self.Encoder_couple = Coupled_Encoder(num_head=4)

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
        image, depth, coord, res, lr_image,edge = data['image'], data['lr'], data['hr_coord'], data['lr_pixel'], data['lr_image'], data['edge']
        lr_guide = self.image_encoder(lr_image)
        feat, hr_guide = self.Encoder_couple(depth, image, edge)

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

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x
