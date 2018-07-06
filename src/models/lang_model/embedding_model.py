import torch.nn as nn
import torch
import torch.autograd as autograd

def tensor_type(use_cuda=False):
    """Required to return the relevant tensor depending on if we use
    cuda or not"""
    if use_cuda:
        float_T = torch.cuda.FloatTensor
        long_T = torch.cuda.LongTensor
    else:
        float_T = torch.FloatTensor
        long_T = torch.LongTensor
    return long_T, float_T

class EmbeddingModel(nn.Module):

    def __init__(self, question_dict, embed_size=50):
        """Module that takes embed dict ({string: iid}) when initialised and
        returns embedding (which can be trained) for that question"""
        super(EmbeddingModel, self).__init__()
        # hidden size for MLP, divide by 2 because muliplied by 2 for other models
        self.num_tokens = int(max(question_dict.values()) + 1)
        self.question_dict = question_dict
        self.embed_layer = torch.nn.Embedding(self.num_tokens, embed_size)
        self.hidden_dim = int(embed_size / 2)
        self.long_T, self.float_T = tensor_type(use_cuda=False)

    def forward(self, sentences):
        """return the mean"""
        # embed these into word vectors
        sentence_ids_list = [self.question_dict[sentence]
                             for sentence in sentences]
        sentence_ids = torch.LongTensor(sentence_ids_list).type(self.long_T)
        sentence_ids.require_grad = True
        sentence_vecs = self.embed_layer(sentence_ids)
        return autograd.Variable(sentence_vecs)

    def reset_parameters(self):
        self.embed_layer.reset_parameters()

    def cuda(self):
        """We need to override this because we need to pass the cuda call
        to the underlying torch object"""
        super(EmbeddingModel, self).cuda()
        self.embed_layer.cuda()
        self.long_T, self.float_T = tensor_type(use_cuda=True)

    def cpu(self):
        """We need to override this because we need to match the input
        data type to the data type of the underlying classifier"""
        super(EmbeddingModel, self).cpu()
        self.embed_layer.cpu()
        self.long_T, self.float_T = tensor_type(use_cuda=False)
