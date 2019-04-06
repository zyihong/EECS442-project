from torchvision import datasets, models, transforms

from model import *


def main():
    net = EncoderNet()

    vgg16 = models.vgg16(pretrained=True)
    net.copy_params_from_vgg16(vgg16)

