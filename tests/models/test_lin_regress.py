from src.models.time_series.ts_models import LinRegress
import torch
import numpy as np

def binary_softmax(vec, temperature=1.):
    return 1 / (1 + torch.exp(-vec/temperature))

def test_lin_regress():

    params = torch.FloatTensor([[3.,7.], [5.,9.]])
    times=np.array([30., 90., 360.])
    lr = LinRegress(times)
    ans = lr.forward(params)

    times = torch.FloatTensor(times)
    b1_ans = 7 + 3*times
    b2_ans = 9 + 5*times
    example_ans = binary_softmax(torch.cat([b1_ans, b2_ans], dim =0).view(2,3))

    assert torch.equal(example_ans, ans)
