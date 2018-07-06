import torch.nn as nn
import torch
import torch.autograd as autograd
from src.data.word2vec import load_w2v_dict
import numpy as np

def preprocessing(string):
    """helper function to remove punctuation froms string"""
    string = string.replace(',', ' ').replace('.', ' ')
    string = string.replace('(', '').replace(')', '')
    words = string.split(' ')
    return words

class Averager(nn.Module):

    def __init__(self, embeddings):
        """module that returns mean word vector for a question using glove embedding"""
        super(Averager, self).__init__()
        # TODO: include torch loading, get rid of gensim
        # TODO: fill in embedding with pytorch glove
        self.embeddings = embeddings
        self.embedding_dim = self.embeddings['the'].shape[0]
        self.use_cuda = False # Initialise on the cpu, call .cuda() for GPU

    def forward(self, sentence):
        """return the mean"""
        # remove punctuation and split into words
        words = preprocessing(sentence)

        # embed these into word vectors
        vecs = [self.embeddings[w] for w in words if w in self.embeddings]

        if len(vecs) == 0:
            to_return = torch.zeros(self.embedding_dim)
        else:
            # vstack list of arrays into array and convert to torch
            vecs_torch = torch.from_numpy(np.vstack(vecs))
            # return the mean
            to_return = torch.mean(vecs_torch.float(), dim=0)
        # put on GPU if necessary
        if self.use_cuda:
            return to_return.cuda()
        else:
            return to_return

    def cuda(self):
        """We need to override this because the conversion from
        numpy matrices to torch tensors is not affected by the
        default .cuda() method"""
        super(Averager, self).cuda()
        self.use_cuda = True

    def cpu(self):
        """We need to override this because the conversion from
        numpy matrices to torch tensors is not affected by the
        default .cuda() or .cpu()  method"""
        super(Averager, self).cpu()
        self.use_cuda = False

class W2vAveragerModel(nn.Module):

    def __init__(self, verbose=True):
        """Module that loads embed dict when initialised and
        returns mean word vector for a question
        using glove embedding"""
        # In future could load the glove embeddings into the standard
        # torch embedding matrix. While this would allow us to fine-tune
        # the embeddings, this doesn't seem likely to be that useful at
        # the moment.
        super(W2vAveragerModel, self).__init__()
        embed_dict = load_w2v_dict(verbose)
        self.averager = Averager(embed_dict)
        # hidden size for MLP, divide by 2 because mulsiplied by 2 for other models
        self.hidden_dim = embed_dict['the'].size/2

    def forward(self, sentences):
        """return the mean"""
        # embed these into word vectors
        sentence_vecs = [self.averager(sentence) for sentence in sentences]
        sentence_vecs = torch.stack(sentence_vecs)
        return autograd.Variable(sentence_vecs)

    def reset_parameters(self):
        pass

    def cuda(self):
        """We need to override this because we need to pass the cuda call
        to the underlying torch object"""
        super(W2vAveragerModel, self).cuda()
        self.averager.cuda()

    def cpu(self):
        """We need to override this because we need to match the input
        data type to the data type of the underlying classifier"""
        super(W2vAveragerModel, self).cpu()
        self.averager.cpu()
