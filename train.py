from torchvision import datasets, models, transforms
from prepare_data import prepare_data
import torch
import os
from dataloader import get_loader
from model import *
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
# matplotlib.use('tkagg')
import cv2
import matplotlib.pyplot as plt
from PIL import Image
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


SAVE_STEP = 2000
MODEL_DIR = 'models/'
EMBED_SIZE = 256
ENCODER_PATH = './models/encoder-4-82000.ckpt'
DECODER_PATH = './models/decoder-4-82000.ckpt'
LOAD_FROM_CHECKPOINT = False
LEARNING_RATE = 5e-4

def sentence(vocab,sampled_ids):
    captions=[]
    for word_id in sampled_ids:
        word = vocab.idx_to_word[word_id]
        captions.append(word)
        # if word == '<end>':
        #     break
    print(captions)


def sample(encoder, decoder, vocab):
    imagelist=['./data/newresized/COCO_train2014_000000000659.jpg','./data/newresized/COCO_train2014_000000000034.jpg', './data/newresized/COCO_train2014_000000000801.jpg']
    for hh in imagelist:
        image = Image.open(hh)
        image_tensor = torch.Tensor(np.asarray(image)).view((1, 224, 224, 3)).to(device)
        # Generate an caption from the image
        encoder.eval()
        feature = encoder(image_tensor)
        sampled_ids = decoder.predict(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx_to_word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        # Print out the image and the generated caption
        print(sentence)
        # image = Image.open(args.image)
        # plt.imshow(np.asarray(image))
        # plt.show()


def main():
    #prepare_data()

    f = open('./data/vocab.pkl', 'rb')
    vocab = pickle.load(f)
    f.close()
    data_loader = get_loader(vocab=vocab, batch_size=5, shuffle=True)

    #Encoder
    encoder = EncoderNet(EMBED_SIZE).to(device)
    vgg16 = models.vgg16(pretrained=True)
    vgg16.cuda()
    encoder.copy_params_from_vgg16(vgg16)

    decoder = DecoderNet(embed_size=EMBED_SIZE, hidden_size=128, vocab_size=len(vocab.word_to_idx)).to(device)

    if LOAD_FROM_CHECKPOINT:
        encoder.load_state_dict(torch.load(ENCODER_PATH))
        decoder.load_state_dict(torch.load(DECODER_PATH))

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())+list(encoder.fc.parameters())
             # +list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    for epoch in range(1):
        for i, (images, captions, lengths) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            captions = captions.to(device)
            sentence(vocab,captions[0].cpu().numpy())
            plt.imshow(images[0].data.cpu().numpy()[:,:,::-1].astype('uint8'))
            plt.show()
            plt.savefig('temp.jpg')
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # sentence(vocab,targets.cpu().numpy())
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            # print('output', outputs.shape)
            # print('caption', captions.shape)
            loss = criterion(outputs, targets)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, 10, i, 10, loss.item(), np.exp(loss.item())))
            if i % 100 == 0 and i != 0:
                sample(encoder, decoder, vocab)

            if (i + 1) % SAVE_STEP == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    MODEL_DIR, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(), os.path.join(
                    MODEL_DIR, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

        # targets=pack_padded_sequence(captions,lengths=lengths,batch_first=True)[0]


if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    main()

