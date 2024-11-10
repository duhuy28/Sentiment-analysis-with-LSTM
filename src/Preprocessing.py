from collections import defaultdict
import numpy as np
import torch


def read_embeddings(embedding_path):
    embeddings = defaultdict(list)
    with open(embedding_path, 'r', encoding='utf-8') as f:
      for line in f:
          tokens = line.rstrip().split()
          embeddings[tokens[0]] = np.array(tokens[1:], dtype='float32')
    return embeddings


def embed_data(sentence):
    # [ToDo]
    embedded_word_list =[]
    for word in sentence:
        if word in embeddings:
            embedded_word_list.append(embeddings[word])
        else:
            embedded_word_list.append(np.zeros(50))
    return  torch.tensor(np.array(embedded_word_list),dtype=torch.float32)

def split_string(text):
  return text.replace(',', ' ').replace('.', ' ').lower().split()


embeddings_path = "../data/glove.6B.50d.txt"  # path to your embedding file
embeddings = read_embeddings(embeddings_path)