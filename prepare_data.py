import nltk
import pickle
import os
from collections import Counter
from pycocotools.coco import COCO
from PIL import Image
from resizeimage import resizeimage
from tqdm import tqdm
from glove import gloveembedding
import numpy as np

CAPTION_PATH = 'data/annotations/captions_train2014.json'
VOCAB_PATH = './data/vocab.pkl'
EMBED_PATH = './data/embed.pkl'

THRESHOLD = 4

IMAGE_DIR = './data/train2014'
RESIZE_IMAGE_DIR = './data/newresized'
IMAGE_SIZE = [224, 224]


class Vocabulary(object):
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.index = 0
        # self.embed=embed
        # self.dict=dict
        self.own=[]

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.index
            self.idx_to_word[self.index] = word
            self.index += 1
            if word in dict:
                self.own.append(embeddic[dict[word]])
            else:
                self.own.append(np.ones((200))*(self.index-1))

    def __len__(self):
        return len(self.word_to_idx)

    def __call__(self, word):
        if word not in self.word_to_idx:
            return self.word_to_idx['<unk>']
        return self.word_to_idx[word]
    def hh(self):
        return np.array(self.own)


def build_vocab(json_path, threshold):
    coco_annotation = COCO(json_path)
    counter = Counter()
    ids = coco_annotation.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco_annotation.anns[id]['caption'])
        # TODO: please master Chao explain these two lines
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for word in words:
        vocab.add_word(word)
    return vocab,vocab.hh()


def resize_image():
    if not os.path.exists(RESIZE_IMAGE_DIR):
        os.makedirs(RESIZE_IMAGE_DIR)

    images = os.listdir(IMAGE_DIR)
    # num_images = len(images)

    for i, image in enumerate(tqdm(images)):
        with open(os.path.join(IMAGE_DIR, image), 'r+b') as f:
            with Image.open(f) as img:
                # img = resize_image(img, IMAGE_SIZE)
                resize_img = resizeimage.resize_cover(img, IMAGE_SIZE, validate=False)
                resize_img.save(os.path.join(RESIZE_IMAGE_DIR, image), img.format)

    print('Finish resizing images')


def prepare_data():
    vocab,ownembed = build_vocab(CAPTION_PATH, THRESHOLD)
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print('Save word vocab to {}'.format(VOCAB_PATH))
    with open(EMBED_PATH,'wb') as f:
        pickle.dump(ownembed,f)
    print('Save word embed to {}'.format(EMBED_PATH))



# resize_image()
# embeddic, dict=gloveembedding()
# prepare_data()


