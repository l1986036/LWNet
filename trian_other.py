import matplotlib.pyplot as plt

from lib.MobileViT.model import mobile_vit_xx_small, mobile_vit_small
from lib.ShuffleNet_V2 import shufflenet_v2_x0_5, ShuffleNetV2
from lib.model_v3 import mobilenet_v3_large
from lib.resnet import ResNet50
from lib.vgg import vgg
from lib.vit_model import vit_base_patch16_224_in21k
from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# 固定随机种子，保证实验结果是可以复现的
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# data_path = "D:\ClassificationLable\Dataset/NCT-CRC-HE-100K_2/NCT-CRC-HE-100K_split000" # todo 数据集路径
data_path = "D:\ClassificationLable\Dataset\Kather_texture_2016_image_tiles_5000\Kather_texture_2016_image_tiles_5000_split"  # todo 数据集路径
# data_path = "D:\ClassificationLable\Dataset\mydataset_split" # todo 数据集路径# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    'model': 'vit_base_patch16_224_in21k',  # 选择预训练模型
    # 'model': "mobile_vit_small",  # 选择预训练模型
    # 'model': 'efficientnet_b3a',  # 选择预训练模型
    # 'model': 'vgg16',  # 选择预训练模型
    # 'model': 'ResNet50',  # 选择预训练模型
    # 'model': 'shufflenet_v2_x0_5',  # 选择预训练模型
    # 'model': 'mobilenet_v3_large',  # 选择预训练模型
    "img_size": 224,  # 图片输入大小
    "train_dir": osp.join(data_path, "train"),  # todo 训练集路径
    "val_dir": osp.join(data_path, "val"),  # todo 验证集路径
    'device': device,  # 设备
    'lr': 1e-3,  # 学习率
    'batch_size': 64,  # 批次大小
    'num_workers': 4,  # 进程
    'epochs': 100,  # 轮数
    "save_dir": "./save_weight/",  # todo 保存路径
    "pretrained": True,
     "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
    'weight_decay': 1e-5  # 学习率衰减
}
result_img = osp.join("D:\ClassificationLable\My_Classification/result_img/", params['model'])  # 设置模型保存路径
if not osp.isdir(result_img):  # 如果保存路径不存在的话就创建
    os.makedirs(result_img)  #
    print("save dir {} created".format(result_img))

# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['num_classes'],
                 pretrained=True):
        super().__init__()
        # self.net = ResNet50()
        # self.net = shufflenet_v2_x0_5()
        # self.net = mobilenet_v3_large()
        # self.net = vgg(model_name=model_name, init_weights=True)
        self.net = vit_base_patch16_224_in21k()


        # self.net = mobile_vit_small()
        model_weight_path = "D:\ClassificationLable\My_Classification\pretrained\jx_vit_base_patch16_224_in21k-e5005f0a.pth"
        assert os.path.exists(model_weight_path), "文件 {}不存在。".format(model_weight_path)
        # new_state_dict = {}
        # model = torch.load(model_weight_path, map_location='cpu')
        # for k_, v in model.items():
        #     if 'fc.weight' not in k_ and 'fc.bias' not in k_:
        #         new_state_dict[k_] = v
        # self.net.load_state_dict(model, strict=False)  # strict=False 表示允许部分加载，即如果模型结构有差异，也能够加载成功。
        self.net.load_state_dict(torch.load(model_weight_path, map_location='cpu'),strict=False)

        for param in self.net.parameters():  # 将所有参数的 requires_grad 设置为 False，这样在后续的训练中，这些参数将不会更新（不进行梯度下降）
            param.requires_grad = False

        # classifier
        # n_features = self.net.head.num_features
        self.net.head = nn.Linear(768, out_features)#你要看self.net有没有head操作
        # n_features = self.net.classifier.last_channel
        # self.net.classifier[0] =  nn.Linear(n_features, out_features)
        # self.net.fc = nn.Linear(n_features, out_features)#你要看self.net有没有head操作

        # self.net.classifier.add_module(name="fc", module=nn.Linear(in_features=1280, out_features=out_features))
        # resnet修改最后的全链接层
        print(self.net)  # 返回模型

    def forward(self, x):  # 前向传播
        x = self.net(x)
        return x

# 定义训练流程
def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()  # 设置指标监视器
    model.train()  # 模型设置为训练模型
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):  # 开始训练
        images = images.to(params['device'], non_blocking=True)  # 加载数据
        target = target.to(params['device'], non_blocking=True)  # 加载模型
        output = model(images)  # 数据送入模型进行前向传播
        loss = criterion(output, target.long())  # 计算损失
        f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
        recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
        acc = accuracy(output, target)  # 计算准确率分数
        metric_monitor.update('Loss', loss.item())  # 更新损失
        metric_monitor.update('F1', f1_macro)  # 更新f1
        metric_monitor.update('Recall', recall_macro)  # 更新recall
        metric_monitor.update('Accuracy', acc)  # 更新准确率
        optimizer.zero_grad()  # 清空学习率
        loss.backward()  # 损失反向传播
        optimizer.step()  # 更新优化器
        # lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)  # 调整学习率
        stream.set_description(  # 更新进度条
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]  # 返回结果


# 定义验证流程
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    with torch.no_grad():  # 开始推理
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)  # 读取图片
            target = target.to(params['device'], non_blocking=True)  # 读取标签
            output = model(images)  # 前向传播
            loss = criterion(output, target.long())  # 计算损失
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc
            metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]


# 展示训练过程的曲线
def show_loss_acc(acc, loss, val_acc, val_loss, sava_dir):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # 保存在savedir目录下。
    save_path = osp.join(result_img, "process.png")
    # plt.show()
    plt.savefig(save_path, dpi=100)
    plt.close()




if __name__ == '__main__':
    accs = []
    losss = []
    val_accs = []
    val_losss = []
    data_transforms = get_torch_transforms(img_size=params["img_size"])  # 获取图像预处理方式
    train_transforms = data_transforms['train']  # 训练集数据处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    train_dataset = datasets.ImageFolder(params["train_dir"], train_transforms)  # 加载训练集
    valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    if params['pretrained'] == True:
        # save_dir = osp.join(params['save_dir'], params['model'])  # 设置模型保存路径
        save_dir = osp.join(params['save_dir'])  # 设置模型保存路径
    else:
        save_dir = osp.join(params['save_dir'], params['model'] + "_nopretrained_" + str(params["img_size"]))  # 设置模型保存路径
    if not osp.isdir(save_dir):  # 如果保存路径不存在的话就创建
        os.makedirs(save_dir)  #
        print("save dir {} created".format(save_dir))
    train_loader = DataLoader(  # 按照批次加载训练集
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(  # 按照批次加载验证集
        valid_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )
    print(train_dataset.classes)
    model = SELFMODEL(model_name=params['model'], out_features=params['num_classes'],
                      pretrained=params['pretrained'])  # 加载模型
    # model = nn.DataParallel(model)  # 模型并行化，提高模型的速度
    # resnet50d_1epochs_accuracy0.50424_weights.pth
    model = model.to(params['device'])  # 模型部署到设备上
    criterion = nn.CrossEntropyLoss().to(params['device'])  # 设置损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])  # 设置优化器

    best_acc = 0.0  # 记录最好的准确率
    # 只保存最好的那个模型。
    i = 1
    for epoch in range(1, params['epochs'] + 1):  # 开始训练
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, params)
        accs.append(acc)
        losss.append(loss)
        val_accs.append(val_acc)
        val_losss.append(val_loss)

        save_path = osp.join(save_dir, f"{params['model']}_{epoch}.pth")
        torch.save(model.state_dict(), save_path)

        if val_acc >= best_acc:
            print("第{}轮{}好".format(epoch, i))
            best_acc = val_acc
            i = i + 1

    show_loss_acc(accs, losss, val_accs, val_losss, save_dir)
    print("训练已完成，模型和训练日志保存在: {}".format(save_dir))


