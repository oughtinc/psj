import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic

class BaselineMF:
    def __init__(self, cf_algo=None, logit=False):
        """
        fit method takes a ContentDataset and fits it for num_epochs (passed at initialisation)

        Parameters
        ----------

        batch_size (int): the size of each training batch

        network (ContentMF): a network that fits using user_ids and item_texts

        num_epochs (int): the number of training epochs

        optim_params (dict): parameters passed to the Stochastic Gradient Descent (SGD) class

        use_cuda (bool): set to True to use the GPU

        """
        self.logit = logit
        self.question_truth_dict = {}
        self.average_true_rating = 0.5
        self.average_false_rating = 0.5
        self.loss_fn = nn.MSELoss(size_average=True)

        if cf_algo is None:
            self.cf_algo = KNNBasic(k=2)
        else:
            self.cf_algo = cf_algo


        #self.svd = SVD(n_epochs=500, verbose=True, lr_all=0.001, n_factors=50)

    def dataloader_extract(self, sample):
        ratings = pd.Series(np.array(list(sample['rating'])))
        user_ids = pd.Series(sample['user_id']).astype(str)
        item_ids = pd.Series(sample['item_id']).astype(str)

        return ratings, user_ids, item_ids

    def logit_fn(self, p, epsilon=1e-3):
        for item in p:
            if item == 0:
                item = epsilon
            if item == 1:
                item = 1 - epsilon
        return np.log(p/(1-p))

    def sigmoid_fn(self, x):
        return 1/(1+np.exp(-x))

    def fit(self, dataset, train_sampler):
        """Runs the fit method which simply works out the average response
        for 'true' and 'false' questions, where 'true' questions are those
        where the average rating is greater than 0.5"""
        t0 = time.time()
        data_loader = DataLoader(dataset, batch_size=len(train_sampler), sampler=train_sampler)
        sample = iter(data_loader).next()
        ratings, user_ids, item_ids = self.dataloader_extract(sample)
        if self.logit:
            ratings = self.logit_fn(ratings)
        possible_ratings = ratings.unique()

        ratings_dict = {'itemID': item_ids,
                        'userID': user_ids,
                        'rating': ratings}
        df = pd.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.cf_algo.train(trainset)


    def predict(self, dataset, sampler, batch_size=64):
        # I'm not entirely sure that the build_full_testset
        # function works as I'd expect, so instead we loop
        # through all the test ids and predict one-at-a-time
        preds = []
        data_loader = DataLoader(dataset, batch_size=len(dataset), sampler=sampler)
        sample = iter(data_loader).next()
        ratings, user_ids, item_ids = self.dataloader_extract(sample)
        for user_id, item_id in zip(user_ids, item_ids):
            pred = self.cf_algo.predict(str(user_id), str(item_id))[3]
            if self.logit:
                pred = self.sigmoid_fn(pred)
            preds.append(pred)



        return(preds)

    def score(self, dataset, sampler, batch_size=64, only_slow=True):
        """Scores the baseline on predictions made on the dataset provided,
        sampled with the given sampler. If `only_slow` is true, then only
        the slow judgments in the sampled part of the dataset are scored"""
        predictions = self.predict(dataset, sampler, batch_size)
        data_loader = DataLoader(dataset, batch_size=len(dataset), sampler=sampler)
        testset = iter(data_loader).next()
        ratings, user_ids, item_ids,  = self.dataloader_extract(testset)
        user_ids = user_ids.astype(int)
        ratings = torch.Tensor(ratings)
        predictions = torch.Tensor(predictions)

        #Note that all baselines are passed flattened datasets, so we
        # have to work out which of the users correspond to the latest
        # times
        if only_slow:
            long_time_uids = [i for i in np.unique(user_ids) if i % 3==2]
            new_ratings = []
            new_preds = []
            for index, rating in enumerate(ratings):
                if user_ids[index] in long_time_uids: new_ratings.append(rating)
            for index, pred in enumerate(predictions):
                if user_ids[index] in long_time_uids: new_preds.append(pred)
            loss = self.loss_fn(torch.Tensor(new_preds), torch.Tensor(new_ratings).cpu())
            return loss.cpu().data.item()

        else:
            loss = self.loss_fn(predictions, ratings.cpu())
            return loss.cpu().data.item()
