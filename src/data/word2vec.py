import os
import time
import zipfile
from definitions import DATA_DIR
import numpy as np

def load_w2v_dict(verbose=True):
    """Returns w2v dict"""
    w2v_filename = 'word2vec.6B.50d.txt'
    w2v_path = os.path.join(DATA_DIR, '{}.zip'.format(w2v_filename))

    t0 = time.time()
    if verbose:
        print("Loading w2v dict")

    with zipfile.ZipFile(w2v_path) as w2v_zip:
        with w2v_zip.open(w2v_filename) as f:
            embeddings = {}
            for line in f.readlines():
                values = line.split()
                word = values[0].decode("utf-8")
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
    if verbose:
        print('Loaded Word2Vec dict: {:0.2f}s'.format(time.time() - t0))
        print('Number of words in corpus: {}'.format(len(embeddings)))
    return embeddings