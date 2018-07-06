import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data.sampler import SubsetRandomSampler
from surprise import KNNBasic, SVD, NormalPredictor

from src.models.content_aware.simple_baseline import BaselineMF
from src.models.content_aware.sampler import SubsetDeterministicSampler
from src.models.collab.two_class_mf import TwoClassMF

def calibration_plot(targets, predictions, weights=None, alpha=0.5, jitter=False, label_string=''):
    import textwrap
    fig, axes = plt.subplots(1, figsize=(12, 6))
    if jitter:
        targets += torch.FloatTensor(np.random.normal(0, 0.005, size=len(targets)))
    axes.scatter(targets, predictions, alpha=alpha)
    axes.set_xlabel('Ground truth')
    axes.set_ylabel('Prediction')
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])
    title_wrapped_str = '\n'.join(textwrap.wrap(label_string, 140))
    title = axes.set_title(title_wrapped_str, fontsize=8)
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)

def run_baseline(train_dataset, val_dataset, baseline_alg_str,
                 calibrate=False, cf_algo_params={}):
    """Given a dataset, run the baseline identified by
    baseline_alg_str on the dataset and return the
    val and train Brier loss. We pass the dictionary of
    parameters in cf_algo_params through to the surprise
    algorithm.

    train_dataset: ContentDataset with the train data
    val_dataset:   ContentDataset with the validation data
    baseline_alg_str:
                      One of 'KNN', 'SVD', 'NormalPredictor',
                      'TwoClassMF', and 'logitSVD'. KNN, SVD and
                      NormalPredictor are as described in the surprise docs,
                      logitSVD is the SVD where the inputs are logit-transformed
                      first, and TwoClassMF splits the questions into true/false
                      based on user majority vote and predicts the mean for true/false
                      , else 50%
    calibrate: whether to plot a calibration curve
    cf_algo_params: passed through to the collaborative filtering
                    algorithm
                      """



    if baseline_alg_str == 'KNN':
        baseline_cf_algo = KNNBasic(**cf_algo_params)
    elif baseline_alg_str == 'SVD':
        baseline_cf_algo = SVD(**cf_algo_params)
    elif baseline_alg_str == 'NormalPredictor':
        baseline_cf_algo = NormalPredictor
    elif baseline_alg_str == 'TwoClassMF':
        baseline_cf_algo = TwoClassMF
    elif baseline_alg_str == 'logitSVD':
        baseline_cf_algo = SVD(**cf_algo_params)

    # Load in data
    baseline = BaselineMF(baseline_cf_algo)
    train_idx = np.arange(len(train_dataset.ratings))
    val_idx = np.arange(len(val_dataset.ratings))

    # Train CF algo and evaluate
    baseline.fit(train_dataset, SubsetRandomSampler(train_idx))
    train_pred = np.array(baseline.predict(
        train_dataset, SubsetDeterministicSampler(train_idx)))
    train_ratings = np.array(train_dataset.ratings)[
        train_idx].reshape((len(train_idx)))
    train_loss = ((train_pred - train_ratings)**2).mean()

    # Evaluate on val
    val_pred = np.array(baseline.predict(
        val_dataset, SubsetDeterministicSampler(val_idx)))
    val_ratings = np.array(val_dataset.ratings)[
        val_idx].reshape((len(val_idx)))
    val_loss = ((val_pred - val_ratings)**2).mean()

    if calibrate:
        calibration_plot(val_ratings, val_pred)
    # Todo change the train methods in baselineMF so that they return the train loss
    return train_loss, val_loss
