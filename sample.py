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

f=open('./data/embed.pkl','rb')
embed=pickle.load(f)
f.close()
decoder = DecoderNet(embed_size=EMBED_SIZE, hidden_size=128, embeddic=embed, vocab_size=len(vocab.word_to_idx)).to(device)

if LOAD_FROM_CHECKPOINT:
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    decoder.load_state_dict(torch.load(DECODER_PATH))


root='./data/resizedTrain2014/'
files=os.listdir(root)
for i in range(1000):
    print (i)
    image = cv2.cvtColor(cv2.imread(os.path.join(root,files[i])),cv2.COLOR_BGR2RGB)
    image_tensor = torch.Tensor(image).view((1, 256, 256, 3)).to(device)
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
    # image = Image.open(args.image)
    plt.imshow(image_tensor[0].data.cpu().numpy().astype('uint8'))
    plt.title(sentence)
    plt.savefig("./results2/img{}.png".format(i))
    # plt.show()

