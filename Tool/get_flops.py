import torch
from torchstat import stat

from lib.ShuffleNet_V2 import shufflenet_v2_x0_5
from lib.mobilenetv3 import mobilenet_v3_large
from lib.mynet.Net1_self_attention_Impro import Net1_ResNet_Impro
from lib.mynet.Net3_Channel_Shufflet_Impro import Net3_Channel_Shufflet_Impro
from lib.resnet import ResNet50
from lib.vgg import vgg
from train import SELFMODEL
from lib.vit_model import vit_base_patch16_224_in21k
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#=======================================================================================
# model_name = 'Net1_ResNet_Impro_10'  # todo 模型名称
# model_path = r"D:\ClassificationLable\My_Classification\save_weight/{}.pth".format(model_name)  # todo 模型地址
# num_classes = 9 # todo 类别数目
# model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
# weights = torch.load(model_path)
# model.load_state_dict(weights)
# model.eval()
# stat(model, (3, 224, 224)) # 后面的224表示模型的输入大小
#=======================================================================================
from thop import profile

def cal_param(net):
    # model = torch.nn.DataParallel(net)
    inputs = torch.randn([1, 3, 224, 224]).cuda()
    flop, para = profile(net, inputs=(inputs,), verbose=False)
    return 'Flops：' + str(2 * flop / 1000 ** 3) + 'G', 'Params：' + str(para / 1000 ** 2) + 'M'


if __name__ == "__main__":
    # net = pre_encoder(8).cuda()
    # net = vgg().cuda()
    # net = ResNet50().cuda()
    # net = Net1_ResNet_Impro().cuda()
    # net = mobilenet_v3_large().cuda()
    # net = shufflenet_v2_x0_5().cuda()
    net = vit_base_patch16_224_in21k().cuda()
    # net = Net3_Channel_Shufflet_Impro().cuda()
    print(cal_param(net))