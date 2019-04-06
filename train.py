from torchvision import datasets, models, transforms
from prepare_data import prepare_data

from model import *


def main():
    prepare_data()

    net = EncoderNet()

    vgg16 = models.vgg16(pretrained=True)
    net.copy_params_from_vgg16(vgg16)

