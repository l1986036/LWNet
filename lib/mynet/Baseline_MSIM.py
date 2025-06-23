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
class MS(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        self.init__ = super(MS, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        ## ---------------------------------------- ##
        self.layer0 = BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv2d(in_channel2, out_channel // 2, 1)

        self.layer3_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),)
        self.layer3_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),)

        self.layer5_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),)
        self.layer5_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),)

        self.layer_out = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channel), act_fn)

    def forward(self, x0, x1):
        ## ------------------------------------------------------------------ ##
        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)

        x_3_1 = self.layer3_1(torch.cat((x0_1, x1_1), dim=1))
        x_5_1 = self.layer5_1(torch.cat((x1_1, x0_1), dim=1))

        x_3_2 = self.layer3_2(torch.cat((x_3_1, x_5_1), dim=1))
        x_5_2 = self.layer5_2(torch.cat((x_5_1, x_3_1), dim=1))
        out = self.layer_out(x_3_2*x0_1 + x_5_2*x1_1)

        return out
class Baseline_MSIM(nn.Module):
    # resnet based encoder decoder
    def __init__(self, num_classes):
        super(Baseline_MSIM, self).__init__()
        self.DWConv = DWConv(3, 24)
        self.high_fusion1 = MS(24, 24, 24)
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
        low_x = x0+ x2
        conv1 = self.layer_edge0(low_x)#48×56×56
        conv2 = self.layer_edge1(conv1)#196×28×28


        ## -------------------------------------- ##
        high_x01 = self.high_fusion1(x1, x2)  # 24×112×112

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
    net = Baseline_MSIM(8).cuda()
    print(cal_param(net))
    # inputs = torch.randn([1, 8, 320, 320])
    # mask = cal_weights(8)
    # print(mask(inputs)[0])

