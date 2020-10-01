# Author: Avrahami (abraham.israeli@post.idc.ac.il), last update: 1.10.2020

import os
import pickle
import numpy as np
import torch 
import scipy.io


def _fetch(path, name):
    """
    an internal function to convert files into tokens and counts
    :param path: str
        the path to the file
    :param name: str
        train/valid/test
    :return: dict
        a dictionary with tokens and counts. In case it is a test dataset, we split it into 2 parts
    """
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    # special case here
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 
                    'tokens_1': tokens_1, 'counts_1': counts_1, 
                        'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}


def get_data(path):
    """
    a function to get the data out of a given path (splitted into train/valid and test).
    Mainly uses the '_fetch' function to retrieve the data
    :param path: str
        the path where all data is located (vocab, train, valid and test)
    :return: tuple
        tuple of size 4 with all relevant information
    """
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    train = _fetch(path, 'train')
    valid = _fetch(path, 'valid')
    test = _fetch(path, 'test')

    return vocab, train, valid, test


def get_batch(tokens, counts, ind, vocab_size, device):
    """
    fetch input data by batch
    :param tokens: list
        the list of tokens in the data
    :param counts: list
        count of each token
    :param ind: list
        indicator list (which docs to use)
    :param vocab_size: int
        size of the vocabulary
    :param device: str
        the device the model runs on
    :return: torch dataframe
        a torch with all required data
    """
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        L = count.shape[1]
        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch
