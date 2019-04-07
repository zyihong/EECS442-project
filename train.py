from torchvision import datasets, models, transforms
from prepare_data import prepare_data
import torch
from dataloader import get_loader
from model import *
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    #prepare_data()

    f=open('./data/vocab.pkl','rb')
    vocab=pickle.load(f)
    f.close()
    data_loader = get_loader(vocab=vocab,batch_size=5)

    #Encoder
    # net = EncoderNet()
    # vgg16 = models.vgg16(pretrained=True)
    # net.copy_params_from_vgg16(vgg16)

    for i, (images,captions,lengths) in enumerate(data_loader):
        images=images.to(device)
        captions=captions.to(device)
        

        # targets=pack_padded_sequence(captions,lengths=lengths,batch_first=True)[0]

    #Decoder








if __name__ == '__main__':
    main()

