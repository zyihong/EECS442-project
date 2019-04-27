from torchvision import datasets, models, transforms
from prepare_data import prepare_data
import torch
import os
from dataloader import get_loader
from model import *
import pickle
import numpy as np
import os
from tqdm import tqdm
import matplotlib
import cv2
import matplotlib.pyplot as plt
import nltk

SAVE_STEP = 500
MODEL_DIR = 'models/'
EMBED_SIZE = 200
ENCODER_PATH = './models/encoder-4-82000.ckpt'
DECODER_PATH = './models/decoder-4-82000.ckpt'
LOAD_FROM_CHECKPOINT = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = EncoderNet(EMBED_SIZE).to(device)
# vgg16 = models.vgg16(pretrained=True)
# vgg16.cuda()
# encoder.copy_params_from_vgg16(vgg16)
encoder.eval()

f = open('./data/vocab.pkl', 'rb')
vocab = pickle.load(f)
f.close()

f = open('./data/embed.pkl','rb')
embed = pickle.load(f)
f.close()
data_loader = get_loader(vocab=vocab, batch_size=5, shuffle=True)
decoder = DecoderNet(embed_size=EMBED_SIZE, hidden_size=128, embeddic=embed, vocab_size=len(vocab.word_to_idx)).to(device)

if LOAD_FROM_CHECKPOINT:
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    decoder.load_state_dict(torch.load(DECODER_PATH))


root = './data/resizedTrain2014/'
files = os.listdir(root)


def getsentence(sampled_ids):
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx_to_word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    return sampled_caption


bleu = 0
for i, (images, captions, lengths) in enumerate(tqdm(data_loader)):
    images = images.to(device)
    captions = captions.data.cpu().numpy()
    feature = encoder(images)
    sampled_ids = decoder.predict(feature).data.cpu().numpy()
    for ii in range(len(sampled_ids)):
        getsentence(captions[ii]), getsentence(sampled_ids[ii])
print(bleu/data_loader.__len__())
