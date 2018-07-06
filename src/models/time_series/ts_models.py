import torch
import torch.nn.functional as F

def binary_softmax(vec, temperature=1.):
    """Higher temperature reduces input to softmax, avoiding saturation, which can help training.
    If temperature is too high, training may be too slow.
    If temperature is too low, training be slow, or return nans as the softmax may get saturated initially."""
    return 1 / (1 + torch.exp(-vec/temperature))

"""
Matrix factorisation
"""
class MatrixFactor:
    def __init__(self, num_outputs, temperature=1.):
        self.param_dim = num_outputs
        self.temperature = temperature

    def forward(self, params):
        ## Matrix factorisation calc happens in the dot product class
        if self.param_dim == 1:
            vec = params.squeeze()
        else:
            vec = params
        return binary_softmax(vec, temperature=self.temperature)

"""
Linear regression model
"""
class LinRegress:
    def __init__(self, times, temperature=1., softmax=True):
        """times (np array): (3,) numpy array of times in seconds. example: np.array([20, 60, 180])"""
        self.times = torch.FloatTensor(times)
        self.param_dim = 2
        self.temperature = temperature
        self.softmax = softmax

    def forward(self, params):
        slope = params[:, 0]
        intercept = params[:, 1]
           # Need to permute so that we can add on the intercept
        rep_int = intercept.repeat(len(self.times), 1).permute(1, 0)
        vec = rep_int + torch.ger(slope, self.times)
        if self.softmax:
            return binary_softmax(vec, temperature=self.temperature)
        else:
            return torch.clamp(vec, 0.1, 0.9)

    def cuda(self):
        self.times = self.times.cuda()

    def cpu(self):
        self.times = self.times.cpu()

"""
Markov Chain model
"""
class MarkovChain:
    def __init__(self, num_states=21):
        # number of states in the markov process. 21 for exact options in thinkagain.
        self.num_states = num_states
        # make opinion_vals the prob values predicted by the markov chain.
        self.opinion_vals = torch.linspace(0, 1, self.num_states)
        # total params is from two trainsition matrices with num_states^2 params
        # and one initial distribution vector with num_states params
        self.param_dim = int(2*(self.num_states**2) + self.num_states)

    def forward(self, params):
        ## Parameter parsing and loading

        # make opinion_vals [bs, num_states, 1] tensor, where for each sample in the batch
        # the opinion_vals are repeated. This is useful for final dot prodcuts.
        opinion_vals = self.opinion_vals.expand(params.shape[0], self.num_states)
        opinion_vals = torch.unsqueeze(opinion_vals, -1)

        # slice parameters into first, second transition matric and initial dist
        slice_idx = int((self.num_states)**2)
        R1_params = params[:,:slice_idx]
        R2_params = params[:,slice_idx:2*slice_idx]
        initial_dist = params[:,2*slice_idx:]
        # make initial_dist a [bs, num_states, 1] tensor
        initial_dist = torch.unsqueeze(initial_dist, -1)
        # make sure it is a distribution (sum to one along each item in batch)
        initial_dist = F.softmax(initial_dist, dim = 1)

        ## Create transition matrices, each [bs, num_states, num_states]
        R1 = self.make_trans_mat(R1_params, self.num_states)
        R2 = self.make_trans_mat(R2_params, self.num_states)

        ## Opinion dynamics from matrix multiplication
        # each is a [bs, num_states, 1]
        opinion_dist1 = torch.matmul(R1, initial_dist)
        opinion_dist2 = torch.matmul(R2, opinion_dist1)

        ## Return mean opinion
        # for each sample in batch, does dot product of distribution with opinion values
        mean_op_0 = torch.matmul(initial_dist.transpose(1, 2),opinion_vals)
        mean_op_1 = torch.matmul(opinion_dist1.transpose(1, 2),opinion_vals)
        mean_op_2 = torch.matmul(opinion_dist2.transpose(1, 2),opinion_vals)

        # return the concatenated opnion sequence
        # [bs, 3] tensor, 3 is for the three times.
        return torch.squeeze(torch.cat([mean_op_0, mean_op_1, mean_op_2], dim=1))

    def cuda(self):
        self.opinion_vals = self.opinion_vals.cuda()

    def cpu(self):
        self.opinion_vals = self.opinion_vals.cpu()

    def make_trans_mat(self, params, matrix_dim):
        ## Reshape and softmax the parameters to make a transition matrix
        unscaled_R = params.view(params.shape[0], matrix_dim, matrix_dim).transpose(1, 2)
        R = F.softmax(unscaled_R, dim=1) # ensures columns add to 1
        return R


class OnlyMLP:
    def __init__(self, temperature=1., softmax=True):
        # return a [bs, 3] vector of predictions
        self.param_dim = 3
        self.temperature = temperature
        self.softmax = softmax

    def forward(self, params):
        # identity function, so that CatMLP contains all function
        vec = params
        if self.softmax:
            return binary_softmax(vec, temperature=self.temperature)
        else:
            return torch.clamp(vec, 0.1, 0.9)            
    def cuda(self):
        pass

    def cpu(self):
        pass
