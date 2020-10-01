# Author: Avrahami (abraham.israeli@post.idc.ac.il), last update: 1.10.2020
import numpy as np
import mimetypes
import gensim


def get_topic_diversity(beta, topk, verbose: True):
    """
    gets a measure about how diverse are the topics
    :param beta: torch of size (topics X vocab_size)
        the distribution over the vocab of each topic
    :param topk: int
        number of words to take from each topic
    :param verbose: bool, default: True
        whether to print on screen the diversity
    :return: float
        the diversity value
    """
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    td = n_unique / (topk * num_topics)
    if verbose:
        print(f'Topic diversity is: {td}')
    return td


def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if len(doc) == 1:
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if len(doc) == 1:
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj


def get_topic_coherence(beta, data, vocab, verbose=True):
    """

    :param beta: torch of size (topics X vocab_size)
        the distribution over the vocab of each topic
    :param data: list
        list of the documents
    :param vocab: list
        vocabulary list
    :param verbose: bool, default: True
        whether to print on screen the coherence
    :return: int
        the coherence value
    """
    D = len(data)
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k)
    tc = np.mean(TC) / counter
    if verbose:
        print('counter: ', counter)
        print('num topics: ', len(TC))
        print('Topic coherence is: {}'.format(tc))
    return tc


def nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy() 
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors


def prepare_embedding_matrix(emb_data_path, emb_size, vocab, random_seed=1984):
    """
    convert an embeddings model into a numpy array. The embeddings model can be a genism one or a txt file with
    all vector values per each word
    :param emb_data_path: str
        the full path to the embeddings model
    :param emb_size: int
        size of the embeddings
    :param vocab: list
        vocabulary list. It is required since if some words do not appear in the embeddings model, we fill them
        up with a random value
    :param random_seed: int
        a random seed value to be used
    :return: numpy array (of size: vocb_size X emb_size)
        the numpy array of the embeddings
    """
    vocab_size = len(vocab)
    mime = mimetypes.guess_type(emb_data_path)
    np.random.seed(random_seed)
    # in case the file type is text - we'll treat it as such
    if mime[0] is not None:
        vectors = {}
        with open(emb_data_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                if word in vocab:
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
        embeddings = np.zeros((vocab_size, emb_size))
        words_found = 0
        for i, word in enumerate(vocab):
            try:
                embeddings[i] = vectors[word]
                words_found += 1
            except KeyError:
                embeddings[i] = np.random.normal(scale=0.6, size=(emb_size,))
    # in case the file type is not text - we'll assume it is a Gensim model, and we'll treat it as such
    else:
        emb_model = gensim.models.Word2Vec.load(emb_data_path)
        embeddings = np.zeros((vocab_size, emb_size))
        words_found = 0
        for i, word in enumerate(vocab):
            try:
                embeddings[i] = emb_model.wv[word]
                words_found += 1
            except KeyError:
                embeddings[i] = np.random.normal(scale=0.6, size=(emb_size,))
    # in both options (text file of a Gensim model, we do the next 2 steps
    return embeddings
