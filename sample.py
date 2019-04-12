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
# matplotlib.use('tkagg')
import cv2
import matplotlib.pyplot as plt
from PIL import Image
SAVE_STEP = 500
MODEL_DIR = 'models/'
EMBED_SIZE = 256
ENCODER_PATH = './models/encoder-1-33500.ckpt'
DECODER_PATH = './models/decoder-1-33500.ckpt'
LOAD_FROM_CHECKPOINT = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = EncoderNet(EMBED_SIZE).to(device)
vgg16 = models.vgg16(pretrained=True)
vgg16.cuda()
encoder.copy_params_from_vgg16(vgg16)

f = open('./data/vocab.pkl', 'rb')
vocab = pickle.load(f)
f.close()


decoder = DecoderNet(embed_size=EMBED_SIZE, hidden_size=128, vocab_size=len(vocab.word_to_idx)).to(device)

if LOAD_FROM_CHECKPOINT:
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    decoder.load_state_dict(torch.load(DECODER_PATH))


root='./data/resizedTrain2014/'
files=os.listdir(root)
for file in files:
    image = Image.open(os.path.join(root,file))
    image_tensor = torch.Tensor(np.asarray(image)).view((1, 256, 256, 3)).to(device)
    # Generate an caption from the image
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