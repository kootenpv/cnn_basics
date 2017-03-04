import just
import numpy as np

import spacy

from gensim.models import Word2Vec


def _get_embedding_size(embeddings):
    if hasattr(embeddings, "vector_size"):
        size = embeddings.vector_size
    elif hasattr(embeddings, "dim"):
        size = embeddings.dim
    else:
        size = embeddings.syn0.shape[1]
    return size


def load_embeddings(path=None):
    path = just.make_path(path)
    binary = path.endswith("gz") or path.endswith("bz2")
    if binary:
        embeddings = Word2Vec.load_word2vec_format(path, binary=True)
    else:
        embeddings = Word2Vec.load_word2vec_format(path, binary=False)
    esize = _get_embedding_size(embeddings)
    return embeddings, esize


def load_nlp():
    return spacy.load("en")


def prep_cnn(sentences, nlp, embeddings, embedding_size=300):
    spacy_docs = [nlp(x) for x in sentences]
    dim = int(np.mean([len(x) for x in spacy_docs]))
    X = np.zeros((len(spacy_docs), dim, embedding_size))
    for i, x in enumerate(spacy_docs):
        embed = [embeddings[y.text] for y in x if y.text in embeddings]
        if embed:
            token_embed = np.array(embed[-dim:])
            X[i, -token_embed.shape[0]:, :] = token_embed
    return X
