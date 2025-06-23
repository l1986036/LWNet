import math
import os
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
# 固定随机种子，保证实验结果是可以复现的
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class global_module(nn.Module):
    def __init__(self, channels, r=4):
        super(global_module, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xg = self.global_att(x)
        out = self.sig(xg)
        return out
class LocalAttention(nn.Module):
    def __init__(self, channels=64, r=4):
        super(LocalAttention, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        wei = self.sig(xl)
        return wei



class ATT(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, in_channel, out_channel):
        super(ATT, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
    def forward(self, x, x_boun_atten):
        out1 = x + x_boun_atten
        out = self.conv1(out1)
        return out



###############################################################################
## 2022/01/03
###############################################################################
class DWConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DWConv, self).__init__()
        self.dw1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.dw3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.dw5 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=5, padding=2, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.downSample = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        dw1 = self.dw1(x)
        dw3 = self.dw3(x)
        dw5 = self.dw5(x)
        dw1 = self.downSample(dw1)
        dw3 = self.downSample(dw3)
        dw5 = self.downSample(dw5)
        return dw1, dw3, dw5
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class GateFusion(nn.Module):
    def __init__(self, in_planes):
        super(GateFusion, self).__init__()
        self.conv3 = BasicConv2d(in_planes // 3, in_planes // 3, 3, stride=1, padding=1)
        self.conv5 = BasicConv2d(in_planes // 3, in_planes // 3, 5, stride=1, padding=2)
        self.conv7 = BasicConv2d(in_planes // 3, in_planes // 3, 7, stride=1, padding=3)
        self.conv1_2 = BasicConv2d(in_planes* 2+1, in_planes, 1)
        self.conv1 = BasicConv2d(in_planes * 2, in_planes, 1)
        # 添加局部注意力模块
        self.local_attn = LocalAttention(in_planes )
        self.global_attn = global_module(in_planes)

    def forward(self, lf, hf):
        # 融合低频和高频特征
        x = self.conv1(torch.cat((lf, hf), dim=1))
        # 分割通道
        xc = torch.chunk(x, 3, dim=1)
        # 应用不同的卷积
        x1 = self.conv3(xc[0])
        x2 = self.conv5(xc[1])
        x3 = self.conv7(xc[2])
        # 合并结果
        x123 = torch.cat((x1, x2,x3), dim=1)

        local_attn = self.local_attn(x123)
        global_attn = self.global_attn(x123)
        x = x *local_attn +x *global_attn
        return x


class Baseline_FFAM(nn.Module):
    # resnet based encoder decoder
    def __init__(self, num_classes):
        super(Baseline_FFAM, self).__init__()
        self.DWConv = DWConv(3, 24)
        self.low_fusion = GateFusion(24)
        ## ---------------------------------------- ##

        ## ---------------------------------------- ##
        self.layer_edge0 = nn.Sequential(
                                         nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=False),
                                         nn.BatchNorm2d(24),
                                         nn.Conv2d(24, 48, kernel_size=1),
                                         nn.BatchNorm2d(48),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2, stride=2))
        self.layer_edge1 = nn.Sequential(
                                         nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
                                         nn.BatchNorm2d(48),
                                         nn.Conv2d(48, 196, kernel_size=1),
                                         nn.BatchNorm2d(196),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2, stride=2))

        self.cat_01 = ATT(24, 48)
        self.cat_11 = ATT(48, 196)
        self.cat_21 = ATT(196, 512)

        ## ---------------------------------------- ##
        self.downSample = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, xx):
        # ---- feature abstraction -----

        x0, x1, x2 = self.DWConv(xx)
        ## -------------------------------------- ##
        low_x = self.low_fusion(x0, x2)  # 24×112×112
        conv1 = self.layer_edge0(low_x)#48×56×56
        conv2 = self.layer_edge1(conv1)#196×28×28


        ## -------------------------------------- ##
        high_x01 = x1+ x2

        cat_out_01 = self.cat_01(high_x01, low_x)
        cat_out_01 = self.downSample(cat_out_01) #48×56×56

        cat_out11 = self.cat_11(cat_out_01, conv1)
        cat_out11 = self.downSample(cat_out11) #196×28×28

        cat_out21 = self.cat_21(cat_out11, conv2)
        cat_out21 = self.downSample(cat_out21) #512×14×14

        x = cat_out21.mean([2, 3])  # global pool
        x = self.fc(x)
        # ---- output ----
        return x


from thop import profile

def cal_param(net):
    # model = torch.nn.DataParallel(net)
    inputs = torch.randn([1, 3, 224, 224]).cuda()
    flop, para = profile(net, inputs=(inputs,), verbose=False)
    return 'Flops：' + str(2 * flop / 1000 ** 3) + 'G', 'Params：' + str(para / 1000 ** 2) + 'M'


if __name__ == "__main__":
    # net = pre_encoder(8).cuda()
    net = Baseline_FFAM(8).cuda()
    print(cal_param(net))
    # inputs = torch.randn([1, 8, 320, 320])
    # mask = cal_weights(8)
    # print(mask(inputs)[0])

