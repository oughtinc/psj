from torch.utils.data.sampler import Sampler


class SubsetDeterministicSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return self.indices.__iter__()

    def __len__(self):
        return len(self.indices)
