import torch
import pickle
import torch
import os
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence
import nltk


class Cocodataset(data.Dataset):
    def __init__(self, json, vocab):

        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.root = "./data/newresized"

    def __getitem__(self, index):
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']

        # get pictures
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = torch.Tensor(cv2.imread(os.path.join(self.root, path)))

        #get caption index
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.vocab("<start>")]
        for token in tokens:
            caption.append(self.vocab(token))
        caption.append(self.vocab("<end>"))

        return image, torch.LongTensor(caption)

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    # mainly for padding, not sure if it works.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    lengths = [len(cap) for cap in captions]
    captions = pad_sequence(captions, batch_first=True)
    return torch.stack(images, 0), captions, lengths


def get_loader(batch_size, vocab, shuffle=True):
    coco = Cocodataset(json='./data/annotations/captions_train2014.json', vocab=vocab)
    return data.DataLoader(coco, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)




        

