import time
import torch
import torch.nn
from policy.models.resnet import resnet18


# input: [batch_size, 1, 96, 160]
# output: [batch_size, 64, 3, 5]
class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet18, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


# Faster and smaller (input: [1, 32, 64])
class ResNet14(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet14, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.layer4 = torch.nn.Sequential()
        self.cnn.output_layer = torch.nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)

# Faster and smaller (input: [B, 1, 24, 40], output: [B, output_dim, 3, 5])
class ResNetMini(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNetMini, self).__init__()
        self.cnn = resnet18(pretrained=False)
        # 单通道输入卷积层
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 移除layer3和layer4以减少下采样
        self.cnn.layer3 = torch.nn.Sequential()
        self.cnn.layer4 = torch.nn.Sequential()
        # 输出层调整 (128->output_dim，保持3x5空间维度)
        self.cnn.output_layer = torch.nn.Conv2d(128, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)

def YopoBackbone(output_dim):
    return ResNet18(output_dim)

def MaskBackbone(output_dim):
    return ResNetMini(output_dim)

if __name__ == '__main__':
    net = YopoBackbone(64, 3)
    input_ = torch.zeros((1, 1, 96, 160))
    start = time.time()
    output = net(input_)
    print(time.time() - start)
