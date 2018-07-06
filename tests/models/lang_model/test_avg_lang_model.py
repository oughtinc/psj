from src.models.lang_model.w2v_averager_model import W2vAveragerModel
import torch
import nose


def test_w2v_averager1():
    # Does the averager give us the right shape output?
    in_sentence = 'This is a test sentence'
    test_avg = W2vAveragerModel()
    out = test_avg([in_sentence])
    nose.tools.assert_equal(out.shape, torch.Size([1, 50]))


def test_w2v_averager2():
    # Does the averager actually average the vectors?
    in_1 = 'hello'
    in_2 = 'world'
    test_avg = W2vAveragerModel()
    avg_out = test_avg([in_1 + ' ' + in_2]).squeeze()
    manual_average = (test_avg([in_1]).squeeze() + test_avg([in_2]).squeeze()) / 2
    for i in range(50):
        nose.tools.assert_almost_equal(float(avg_out[i]),
                                       float(manual_average[i]))
