import torch
from prettytable import PrettyTable
from thop import profile
from torch import nn, Tensor
from torch.backends import cudnn

from utils.adder import Adder


class CAM(nn.Module):
    """Channel Attention Block for Convolutional Block Attention Module(CBAM)"""

    def __init__(self, base_channel: int, reduction: int = 16):
        super(CAM, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=base_channel, out_channels=base_channel // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=base_channel // reduction, out_channels=base_channel, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_vec = self.fc(self.max_pool(x))
        avg_vec = self.fc(self.avg_pool(x))
        attention = self.sigmoid(max_vec + avg_vec)
        return x * attention


class SAM(nn.Module):
    """Spatial Attention Module for Convolutional Block Attention Module(CBAM)"""

    def __init__(self):
        super(SAM, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        attention = self.conv(torch.cat((avg_pool, max_pool), dim=1))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module(CBAM)
    paper: CBAM: Convolutional Block Attention Module (arXiv: https://arxiv.org/pdf/1807.06521.pdf)
    """

    def __init__(self, base_channel: int):
        super(CBAM, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channel),
            nn.GELU(),
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channel),
        )

        self.channel_att = CAM(base_channel)
        self.spatial_att = SAM()

        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv(x)
        y = self.channel_att(y)
        y = self.spatial_att(y)
        return self.act(x + y)


class ResBlock(nn.Module):
    def __init__(self, base_channels: int):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
        )

    def forward(self, x):
        return x + self.conv(x)


class DenseColorNet(nn.Module):
    def __init__(self, base_channel: int = 32):
        super(DenseColorNet, self).__init__()

        self.gray_ext = nn.Sequential(
            nn.Conv2d(1, base_channel, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.hint_ext = nn.Sequential(
            nn.Conv2d(3, base_channel, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.conv = nn.ModuleList([
            ResBlock(base_channel),
            nn.Sequential(
                CBAM(base_channel * 2),
                CBAM(base_channel * 2),
                CBAM(base_channel * 2),
                CBAM(base_channel * 2),
            ),
            nn.Sequential(
                ResBlock(base_channel),
                ResBlock(base_channel),
                nn.Conv2d(base_channel, 2, kernel_size=1, stride=1, padding=0)
            )
        ])

        self.down12 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(base_channel * 4, base_channel * 2, kernel_size=1),
            nn.GELU()
        )

        self.up21 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(base_channel // 2, base_channel, kernel_size=1),
            nn.GELU()
        )

        self.tanh = nn.Tanh()

    def forward(self, gray, hint):
        x = self.gray_ext(gray)
        y = self.hint_ext(hint)

        res1 = self.conv[0](x + y)
        x = self.down12(res1)

        x = self.conv[1](x)

        x = self.up21(x) + res1
        x = self.conv[2](x)

        return self.tanh(x)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    current_name = ''
    current_params_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.split(sep='.')[0] != current_name:
            table.add_row([current_name, current_params_count])
            current_name = name.split(sep='.')[0]
            current_params_count = 0

        params = parameter.numel()
        table.add_row([name, params])

        total_params += params
        current_params_count += params

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def calc_flops():
    model = DenseColorNet()
    x: Tensor = torch.randn(1, 1, 256, 256).to('cuda')
    hint: Tensor = torch.randn(1, 3, 256, 256).to('cuda')
    macs, params = profile(model.to('cuda'), inputs=(x, hint,))
    print("MACs(G): ", macs / (1000 ** 3), ", Params(M): ", params / (1000 ** 2))


def calc_time():
    cudnn.benchmark = True

    model = DenseColorNet()
    x: Tensor = torch.randn(1, 1, 256, 256, device='cuda')
    hint: Tensor = torch.randn(1, 3, 256, 256, device='cuda')
    model.to('cuda')
    model.eval()

    torch.cuda.empty_cache()

    print('model gpu usage:', torch.cuda.memory_allocated() / 1024 / 1024, "MiB")

    adder = Adder()

    with torch.no_grad():
        for i in range(10):
            _ = model(x, hint)

        for i in range(100):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            starter.record()

            _ = model(x, hint)

            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender)
            adder(elapsed)

    print(adder.average())


if __name__ == '__main__':
    calc_flops()


def create_colorization_model():
    return DenseColorNet()


def create_model():
    model = DenseColorNet()
    count_parameters(model)
    return model
