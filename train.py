from torchvision import datasets, models, transforms
from prepare_data import prepare_data
import torch
from dataloader import get_loader
from model import *
import pickle
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    #prepare_data()

    f = open('./data/vocab.pkl', 'rb')
    vocab = pickle.load(f)
    f.close()
    data_loader = get_loader(vocab=vocab, batch_size=5, shuffle=True)

    #Encoder
    encoder = EncoderNet(50)
    vgg16 = models.vgg16(pretrained=True)
    encoder.copy_params_from_vgg16(vgg16)

    decoder = DecoderNet(embed_size=50, hidden_size=100, vocab_size=data_loader.__len__())

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())+list(encoder.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    for epoch in range(10):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            # print('output', outputs.shape)
            # print('caption', captions.shape)
            loss = criterion(outputs, captions)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, 10, i, 10, loss.item(), np.exp(loss.item())))
        # targets=pack_padded_sequence(captions,lengths=lengths,batch_first=True)[0]


if __name__ == '__main__':
    main()

