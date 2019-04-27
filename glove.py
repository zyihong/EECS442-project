import numpy as np


def gloveembedding():
    dict = {}
    embeddic = []
    f = open("./data/glove.6B/glove.6B.200d.txt", 'r', encoding='UTF-8')
    for i, line in enumerate(f):
        splitLine = line.split()
        word = splitLine[0]
        embeddic.append(np.array([float(val) for val in splitLine[1:]]))
        dict[word] = i
    embeddic = np.array(embeddic)
    print("Done.", len(dict), " words loaded!")
    return embeddic, dict
